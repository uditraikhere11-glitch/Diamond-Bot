"""
Trading Signal Execution Client — Standalone WebSocket Consumer.

Connects to the FastAPI broadcaster server, receives F&O trade signals,
and executes them via Zerodha Kite Connect or Dhan APIs.

Usage:
    python client.py                    # Uses .env in current directory
    python client.py /path/to/.env      # Uses custom env file

Broker selection via BROKER env var:
    BROKER=zerodha   → Zerodha Kite Connect (default)
    BROKER=dhan      → Dhan HQ

Intended to run on:
    - User's local PC (for static IP compliance with SEBI/Zerodha)
    - GCP Compute Engine with a static external IP
"""

import os
import sys
import io
import math
import csv
import json
import uuid
import asyncio
import logging
import time as time_mod
import requests
from datetime import datetime, time, timedelta, timezone
from client_agent.notifier import Notifier

try:  # imported as client_agent.client (tests, fno_agent)
    from client_agent.dhan_token import fresh_token, is_token_error
    from client_agent.equity_protection import ProtectionWatcher
except ImportError:  # run as `python client.py` from client_agent/
    from dhan_token import fresh_token, is_token_error
    from equity_protection import ProtectionWatcher

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("client_agent")

# --- Load Environment ---
try:
    from dotenv import load_dotenv
except ImportError:
    logger.error("python-dotenv is required. Install: pip install python-dotenv")
    sys.exit(1)

env_file = sys.argv[1] if len(sys.argv) > 1 else ".env"
load_dotenv(env_file, override=True)
logger.info(f"Loaded configuration from {env_file}")

# --- Configuration ---
WS_SERVER_URL = os.environ.get("WS_SERVER_URL", "ws://localhost:8000/ws")
WS_AUTH_TOKEN = os.environ.get("WS_AUTH_TOKEN", "")

# Telegram credentials for Notifier
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")

# Broker selection: "zerodha" or "dhan"
BROKER = os.environ.get("BROKER", "zerodha").lower().strip()

# Zerodha credentials
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

# Dhan credentials
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
DHAN_PIN = os.environ.get("DHAN_PIN", "")
DHAN_TOTP_SECRET = os.environ.get("DHAN_TOTP_SECRET", "")

# Execution filters
_RAW_ALLOWED_CHANNELS = os.environ.get("ALLOWED_CHANNELS", "").strip()
ALLOWED_CHANNELS = {c.strip().lower() for c in _RAW_ALLOWED_CHANNELS.split(",")} if _RAW_ALLOWED_CHANNELS else set()

# Capital Allocation
TRADE_AMOUNT_INR = int(os.environ.get("TRADE_AMOUNT_INR", "0"))
DEFAULT_QUANTITY = int(os.environ.get("DEFAULT_QUANTITY", "1"))

# Optional stop-loss + target, as fixed % of the EXECUTED price (placed only after the buy executes).
EQUITY_SL_TARGET_ENABLED = os.environ.get("EQUITY_SL_TARGET_ENABLED", "false").strip().lower() in ("true", "1", "yes")
EQUITY_SL_PCT = float(os.environ.get("EQUITY_SL_PCT", "0") or 0)
EQUITY_TARGET_PCT = float(os.environ.get("EQUITY_TARGET_PCT", "1") or 0)
EQUITY_SL_LIMIT_BUFFER_PCT = float(os.environ.get("EQUITY_SL_LIMIT_BUFFER_PCT", "1") or 0)

# Reconnection settings
RECONNECT_BASE_DELAY = 1.0    # seconds
RECONNECT_MAX_DELAY = 60.0    # seconds
RECONNECT_MULTIPLIER = 2.0


class InstrumentMap:
    def __init__(self, csv_path: str = "instruments.csv"):
        self._map: dict[str, int] = {}
        self._csv_path = csv_path
        self._loaded = False

    def load(self):
        if not os.path.exists(self._csv_path):
            logger.warning(
                f"Instruments file '{self._csv_path}' not found. "
                f"Download from https://api.kite.trade/instruments"
            )
            return

        import csv
        count = 0
        with open(self._csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['exchange']}:{row['tradingsymbol']}"
                self._map[key] = int(row["instrument_token"])
                count += 1

        self._loaded = True
        logger.info(f"Loaded {count} instruments into memory.")

    def get_token(self, exchange: str, tradingsymbol: str) -> int | None:
        key = f"{exchange}:{tradingsymbol}"
        return self._map.get(key)

    def build_tradingsymbol(self, symbol: str, strike: int, option_type: str, expiry: str = "") -> str:
        if expiry:
            return f"{symbol}{expiry}{strike}{option_type}"
        return f"{symbol}{strike}{option_type}"


class ZerodhaBroker:
    def __init__(self, api_key: str, access_token: str, dry_run: bool = True):
        self.api_key = api_key
        self.access_token = access_token
        self.dry_run = dry_run
        self._kite = None
        self._instruments = InstrumentMap()

    def connect(self):
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self.api_key)
            self._kite.set_access_token(self.access_token)
            logger.info("Kite Connect client initialized.")
        except ImportError:
            logger.warning("kiteconnect SDK not installed. Running in simulation mode.")
            self._kite = None
        self._instruments.load()

    def place_order(self, signal: dict) -> dict | None:
        action = signal["action"]
        symbol = signal["symbol"]
        strike = signal.get("strike", 0)
        option_type = signal.get("option_type", "CE")
        order_type = signal.get("order_type", "MARKET")
        entry_price = signal.get("entry_price")

        tradingsymbol = self._instruments.build_tradingsymbol(symbol, strike, option_type)
        exchange = "NFO"
        token = self._instruments.get_token(exchange, tradingsymbol)
        transaction_type = "BUY" if action == "BUY" else "SELL"

        quantity = DEFAULT_QUANTITY
        if TRADE_AMOUNT_INR > 0:
            if order_type == "LIMIT" and entry_price and entry_price > 0:
                quantity = max(1, math.floor(TRADE_AMOUNT_INR / entry_price))
            else:
                try:
                    if self._kite is not None:
                        quote_key = f"{exchange}:{tradingsymbol}"
                        t0 = time_mod.perf_counter()
                        quote = self._kite.quote([quote_key])
                        t1 = time_mod.perf_counter()
                        ltp = quote.get(quote_key, {}).get("last_price", 0)
                        if ltp and ltp > 0:
                            quantity = max(1, math.floor(TRADE_AMOUNT_INR / ltp))
                except Exception as e:
                    logger.error(f"Failed to fetch live price via Kite: {e}")

        order_params = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": "MIS",
            "order_type": order_type,
            "validity": "DAY",
        }

        if order_type == "LIMIT" and entry_price is not None:
            order_params["price"] = entry_price

        if self.dry_run or self._kite is None:
            return {"order_id": "DRY_RUN", "status": "SIMULATED", "params": order_params}

        try:
            order_id = self._kite.place_order(variety="regular", **order_params)
            return {"order_id": order_id, "status": "PLACED"}
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"status": "REJECTED", "response": str(e)}

    def exit_position(self, signal: dict) -> dict | None:
        symbol = signal["symbol"]
        strike = signal.get("strike", 0)
        option_type = signal.get("option_type", "CE")
        tradingsymbol = self._instruments.build_tradingsymbol(symbol, strike, option_type)
        
        exit_params = {
            "tradingsymbol": tradingsymbol,
            "exchange": "NFO",
            "transaction_type": "SELL",
            "quantity": 1,
            "product": "MIS",
            "order_type": "MARKET",
            "validity": "DAY",
        }
        if self.dry_run or self._kite is None:
            return {"order_id": "DRY_RUN_EXIT", "status": "SIMULATED", "params": exit_params}
        try:
            order_id = self._kite.place_order(variety="regular", **exit_params)
            return {"order_id": order_id, "status": "EXITED"}
        except Exception as e:
            return {"status": "REJECTED", "response": str(e)}

    def handle_signal(self, signal: dict) -> dict | None:
        action = signal.get("action", "").upper()
        if action in ("BUY", "SELL"):
            return self.place_order(signal)
        elif action == "EXIT":
            return self.exit_position(signal)
        elif action == "UPDATE":
            return {"status": "ACKNOWLEDGED", "action": "UPDATE"}
        return None


class DhanScripMap:
    SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

    def __init__(self):
        self._equity_map: dict[str, dict] = {}
        self._fno_map: dict[str, dict] = {}
        self._loaded = False

    def load(self):
        logger.info("Downloading Dhan scrip master CSV...")
        try:
            response = requests.get(self.SCRIP_MASTER_URL, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download Dhan scrip master: HTTP {response.status_code}")
                return

            csv_data = csv.DictReader(io.StringIO(response.text))
            for row in csv_data:
                exchange = row.get("SEM_EXM_EXCH_ID", "")
                inst_type = row.get("SEM_INSTRUMENT_NAME", "")
                symbol = row.get("SEM_TRADING_SYMBOL", "")
                sec_id = row.get("SEM_SMST_SECURITY_ID", "")

                if not symbol or not sec_id:
                    continue

                if exchange in ("NSE", "BSE") and inst_type == "EQUITY":
                    clean_symbol = symbol.replace("-EQ", "").strip()
                    segment = "NSE_EQ" if exchange == "NSE" else "BSE_EQ"
                    if clean_symbol not in self._equity_map or segment == "NSE_EQ":
                        self._equity_map[clean_symbol] = {"id": sec_id, "segment": segment}

            self._loaded = True
            logger.info(f"Loaded {len(self._equity_map)} equities from Dhan.")
        except Exception as e:
            logger.error(f"Error loading Dhan scrip master: {e}")

    def get_equity(self, symbol: str) -> dict | None:
        return self._equity_map.get(symbol.upper())


def _is_amo_window() -> bool:
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return True
    return now.time() < time(9, 15) or now.time() >= time(15, 15)


class DhanBroker:
    ORDERS_URL = "https://api.dhan.co/v2/orders"
    PROFILE_URL = "https://api.dhan.co/v2/profile"

    def __init__(self, client_id: str, access_token: str, pin: str = "", totp_secret: str = "", dry_run: bool = True):
        self.client_id = client_id
        self.access_token = access_token
        self.pin = pin
        self.totp_secret = totp_secret
        self.dry_run = dry_run
        self._scrips = DhanScripMap()
        self.watcher = None

    def connect(self):
        if not self.access_token and self.pin and self.totp_secret:
            success = self._auto_login(rejected="")
            if success and not self._token_works():
                success = self._auto_login(rejected=self.access_token)
            if not success:
                logger.error("Initial Dhan auto-login failed. Exiting.")
                sys.exit(1)
        self._scrips.load()

    def _token_works(self) -> bool:
        try:
            resp = requests.get(self.PROFILE_URL, headers=self._build_headers(), timeout=10)
            return resp.status_code == 200
        except Exception:
            return True

    def _auto_login(self, rejected: str = "") -> bool:
        token = fresh_token(rejected, self._generate_token)
        if token:
            self.access_token = token
            return True
        return False

    def _generate_token(self) -> str | None:
        try:
            import pyotp
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()
            url = f"https://auth.dhan.co/app/generateAccessToken?dhanClientId={self.client_id}&pin={self.pin}&totp={current_totp}"
            response = requests.post(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "accessToken" in data:
                    return data["accessToken"]
        except Exception as e:
            logger.error(f"Dhan auto-login error: {e}")
        return None

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "access-token": self.access_token,
        }

    def _place_dhan_order(self, payload: dict, is_retry: bool = False, url: str | None = None) -> dict | None:
        url = url or self.ORDERS_URL
        if self.dry_run:
            return {"order_id": "DRY_RUN", "status": "SIMULATED", "params": payload}

        try:
            response = requests.post(url, headers=self._build_headers(), json=payload, timeout=10)
            result = response.json()

            if response.status_code == 200 and "orderId" in result:
                return {"order_id": result["orderId"], "status": result.get("orderStatus", "QUEUED")}
            
            elif is_token_error(response.status_code, result):
                if not is_retry and self.pin and self.totp_secret:
                    if self._auto_login(rejected=self.access_token):
                        return self._place_dhan_order(payload, is_retry=True, url=url)
                return {"status": "REJECTED_UNAUTHORIZED", "response": result}
            else:
                return {"status": "REJECTED", "response": result}
        except Exception as e:
            return {"status": "REJECTED", "response": str(e)}

    def place_order(self, signal: dict) -> dict | None:
        stock_symbol = signal.get("stock_symbol", "")
        order_type = signal.get("order_type", "MARKET")
        entry_price = signal.get("entry_price", 0)

        scrip_info = self._scrips.get_equity(stock_symbol)
        if not scrip_info:
            return {"status": "REJECTED", "response": {"errorMessage": f"Scrip not found for '{stock_symbol}'"}}

        scrip_id = scrip_info["id"]
        segment = scrip_info["segment"]
        use_amo = _is_amo_window()

        price = round((entry_price * 1.01) / 0.05) * 0.05 if order_type == "LIMIT" and entry_price > 0 else 0.0
        
        ltp = None
        if order_type == "MARKET" and TRADE_AMOUNT_INR > 0:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_symbol}.NS?interval=1d&range=1d"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                if resp.status_code == 200:
                    ltp = resp.json().get("chart", {}).get("result", [{}])[0].get("meta", {}).get("regularMarketPrice", 0)
            except Exception:
                pass

        quantity = DEFAULT_QUANTITY
        if TRADE_AMOUNT_INR > 0:
            if order_type == "LIMIT" and entry_price > 0:
                quantity = max(1, math.floor(TRADE_AMOUNT_INR / entry_price))
            elif ltp and ltp > 0:
                quantity = max(1, math.floor(TRADE_AMOUNT_INR / ltp))

        payload = {
            "dhanClientId": self.client_id,
            "correlationId": str(uuid.uuid4())[:30],
            "transactionType": "BUY",
            "exchangeSegment": segment,
            "productType": "CNC",
            "orderType": "MARKET" if price == 0.0 else "LIMIT",
            "validity": "DAY",
            "securityId": str(scrip_id),
            "quantity": quantity,
            "price": price,
            "disclosedQuantity": 0,
            "triggerPrice": 0.0,
            "afterMarketOrder": use_amo,
        }
        if use_amo:
            payload["amoTime"] = "OPEN"

        return self._place_dhan_order(payload)

    def handle_signal(self, signal: dict) -> dict | None:
        return self.place_order(signal)


def _create_broker(dry_run: bool):
    if BROKER == "dhan":
        if not DHAN_CLIENT_ID:
            logger.error("Dhan credentials missing.")
            sys.exit(1)
        broker = DhanBroker(client_id=DHAN_CLIENT_ID, access_token=DHAN_ACCESS_TOKEN, pin=DHAN_PIN, totp_secret=DHAN_TOTP_SECRET, dry_run=dry_run)
        broker.connect()
        return broker
    else:
        broker = ZerodhaBroker(api_key=KITE_API_KEY, access_token=KITE_ACCESS_TOKEN, dry_run=dry_run)
        broker.connect()
        return broker


async def connect_and_listen():
    try:
        import websockets
    except ImportError:
        logger.error("websockets is required. Install: pip install websockets")
        return

    # Initialize Notifier
    notifier = Notifier(TG_BOT_TOKEN, TG_CHAT_ID)
    await notifier.send("🚀 [Equity] Equity bot started successfully.")

    dry_run = os.environ.get("DRY_RUN", "true").lower() in ("true", "1", "yes")
    broker = _create_broker(dry_run)

    delay = RECONNECT_BASE_DELAY

    while True:
        try:
            connect_url = WS_SERVER_URL
            if WS_AUTH_TOKEN:
                separator = "&" if "?" in connect_url else "?"
                connect_url = f"{connect_url}{separator}token={WS_AUTH_TOKEN}"

            async with websockets.connect(connect_url, ping_interval=20, ping_timeout=10) as ws:
                logger.info("Connected to broadcaster server.")
                delay = RECONNECT_BASE_DELAY

                async for raw_message in ws:
                    try:
                        signal = json.loads(raw_message)
                        if signal.get("asset_class") == "FNO":
                            continue

                        channel_label = signal.get("channel_label", "").lower()
                        if ALLOWED_CHANNELS and channel_label not in ALLOWED_CHANNELS:
                            continue

                        stock_symbol = signal.get("stock_symbol", "UNKNOWN")
                        logger.info(f"Processing equity signal for {stock_symbol}")

                        result = broker.handle_signal(signal)

                        if result:
                            status = result.get("status")
                            response = result.get("response", {})
                            if status in ("REJECTED", "REJECTED_UNAUTHORIZED"):
                                err_msg = response.get("errorMessage", str(response))
                                await notifier.send(f"🚨 [Equity] Order Rejected for {stock_symbol}: {err_msg}")
                            elif status in ("SUCCESS", "FILLED", "PENDING", "QUEUED"):
                                await notifier.send(f"✅ [Equity] Order Active/Filled for {stock_symbol} (Status: {status})")
                        else:
                            await notifier.send(f"⚠️ [Equity] No result returned for {stock_symbol}")

                    except Exception as e:
                        logger.error(f"Error processing signal: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await asyncio.sleep(delay)
            delay = min(delay * RECONNECT_MULTIPLIER, RECONNECT_MAX_DELAY)


if __name__ == "__main__":
    try:
        asyncio.run(connect_and_listen())
    except KeyboardInterrupt:
        logger.info("Client stopped by user.")
