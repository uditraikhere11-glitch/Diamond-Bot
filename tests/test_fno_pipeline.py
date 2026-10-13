"""Tests for the FnO pipeline: parser, contract resolution, sizing, executor, and broadcaster routing."""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from fno_agent import journal as J
from fno_agent.brokers.base import CANCELLED, COMPLETE, OPEN, FnOBroker, OrderState
from fno_agent.executor import FnOExecutor
from fno_agent.instruments import ContractNotFound, InstrumentResolver
from fno_agent.market import IST, is_market_open, round_to_tick
from fno_agent.notifier import Notifier
from fno_agent.settings import Settings
from fno_agent.sizer import size_position, split_tranches
from services.fno_parser import parse_fno_regex

POLYCAB = """#POLYCAB 8000 PE OCT @90-105

SL-30

Target-500,1000

Bigger Targets you may need to hold till month end so plan accordingly with appropriate qty

present low liquidity in options so add slowly"""

LT = """#LT 3900PE @80

SL-55

Target-140"""

SENSEX = """Sensex 74500CE@ 450-460

SL-360

Target-820,1100"""

CDSL = """CDSL 1400CE OCT @33-35

SL_10

Target-100+

3 weeks holding

Go with limited qty if any chance we may need to avg"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_polycab():
    s = parse_fno_regex(POLYCAB)
    assert (s.symbol, s.strike, s.option_type, s.expiry_month) == ("POLYCAB", 8000, "PE", "OCT")
    assert (s.entry_min, s.entry_max, s.stop_loss, s.targets) == (90, 105, 30, [500, 1000])
    assert s.caution is True


def test_parse_lt_single_price():
    s = parse_fno_regex(LT)
    assert (s.symbol, s.strike, s.option_type, s.expiry_month) == ("LT", 3900, "PE", None)
    assert (s.entry_min, s.entry_max, s.stop_loss, s.targets) == (80, 80, 55, [140])


def test_parse_sensex_no_space():
    s = parse_fno_regex(SENSEX)
    assert (s.symbol, s.strike, s.option_type) == ("SENSEX", 74500, "CE")
    assert (s.entry_min, s.entry_max, s.stop_loss, s.targets) == (450, 460, 360, [820, 1100])


def test_parse_cdsl_underscore_sl_and_plus_target():
    s = parse_fno_regex(CDSL)
    assert (s.symbol, s.strike, s.option_type, s.expiry_month) == ("CDSL", 1400, "CE", "OCT")
    assert (s.entry_min, s.entry_max, s.stop_loss, s.targets) == (33, 35, 10, [100])
    assert s.caution is True
    assert s.hold_hint == "3 weeks"


def test_parse_symbol_with_ampersand_day_month_and_slash_targets():
    s = parse_fno_regex("#M&M 3200 CE 30 OCT @ 50 to 55\nSL 40\nTarget 80/100")
    assert (s.symbol, s.expiry_day, s.expiry_month, s.entry_min, s.entry_max) == ("M&M", 30, "OCT", 50, 55)
    assert s.targets == [80, 100]


@pytest.mark.parametrize("text", [
    ".",
    "",
    "Book profits in POLYCAB 8000 PE @300",
    "Exit LT 3900PE @120",
    "#RELIANCE @2900 SL 2800",  # equity call
])
def test_parse_rejects(text):
    assert parse_fno_regex(text) is None


@pytest.mark.parametrize("text, sl, targets", [
    ("#LT 3900PE @80", None, []),  # no SL/target at all
    ("#LT 3900PE @80\nTarget-140", None, [140]),  # no SL
    ("#LT 3900PE @80\nSL-90\nTarget-140", None, [140]),  # nonsensical SL dropped, not rejected
    ("#LT 3900PE @80\nSL-55\nTarget-70", 55, []),  # target below entry dropped
])
def test_parse_accepts_signals_without_usable_sl_target(text, sl, targets):
    s = parse_fno_regex(text)
    assert (s.symbol, s.strike, s.entry_max) == ("LT", 3900, 80)
    assert (s.stop_loss, s.targets) == (sl, targets)


# ---------------------------------------------------------------------------
# Contract resolution
# ---------------------------------------------------------------------------

TODAY = date(2026, 9, 23)


def _row(symbol, strike, opt, expiry, exchange="NFO", lot=125):
    return dict(symbol=symbol, strike=float(strike), option_type=opt, expiry=expiry, exchange=exchange,
                lot_size=lot, tick_size=0.05, security_id=f"{symbol}{strike}{opt}{expiry:%m%d}",
                exchange_segment="NSE_FNO" if exchange == "NFO" else "BSE_FNO",
                tradingsymbol="", instrument_token=0)


@pytest.fixture
def resolver():
    rows = [
        _row("POLYCAB", 8000, "PE", date(2026, 9, 29)),
        _row("POLYCAB", 8000, "PE", date(2026, 10, 27)),
        _row("POLYCAB", 8000, "PE", date(2026, 11, 24)),
        _row("LT", 3900, "PE", date(2026, 9, 23)),  # expires today -> skipped by min-days rule
        _row("LT", 3900, "PE", date(2026, 10, 27)),
        _row("SENSEX", 74500, "CE", date(2026, 9, 24), "BFO", 20),
        _row("SENSEX", 74500, "CE", date(2026, 10, 1), "BFO", 20),
        _row("CDSL", 1400, "CE", date(2027, 1, 26), lot=475),
        # Weekly + monthly in the same month for an index
        _row("NIFTY", 25000, "CE", date(2026, 10, 6), lot=75),
        _row("NIFTY", 25000, "CE", date(2026, 10, 27), lot=75),
    ]
    r = InstrumentResolver("dhan", min_days_to_expiry=1)
    r.load(pd.DataFrame(rows), TODAY)
    return r


def test_resolve_named_month_takes_monthly(resolver):
    c = resolver.resolve("POLYCAB", 8000, "PE", "OCT", today=TODAY)
    assert c.expiry == date(2026, 10, 27) and c.lot_size == 125 and c.exchange_segment == "NSE_FNO"
    assert resolver.resolve("NIFTY", 25000, "CE", "OCT", today=TODAY).expiry == date(2026, 10, 27)


def test_resolve_no_month_nearest_respecting_min_days(resolver):
    assert resolver.resolve("POLYCAB", 8000, "PE", today=TODAY).expiry == date(2026, 9, 29)
    assert resolver.resolve("LT", 3900, "PE", today=TODAY).expiry == date(2026, 10, 27)


def test_resolve_sensex_routes_to_bfo(resolver):
    c = resolver.resolve("SENSEX", 74500, "CE", today=TODAY)
    assert c.exchange == "BFO" and c.expiry == date(2026, 9, 24) and c.lot_size == 20


def test_resolve_past_month_rolls_to_next_year(resolver):
    assert resolver.resolve("CDSL", 1400, "CE", "JAN", today=TODAY).expiry == date(2027, 1, 26)


def test_resolve_missing_month_or_strike_is_rejected(resolver):
    with pytest.raises(ContractNotFound):
        resolver.resolve("POLYCAB", 8000, "PE", "DEC", today=TODAY)  # no silent fallback
    with pytest.raises(ContractNotFound):
        resolver.resolve("POLYCAB", 8100, "PE", today=TODAY)


# ---------------------------------------------------------------------------
# Sizing / market helpers
# ---------------------------------------------------------------------------

def test_size_position():
    assert size_position(50000, 125, 105).lots == 3  # 13,125 per lot
    assert size_position(50000, 125, 105).qty == 375
    assert size_position(10000, 125, 105).lots == 0  # 1 lot exceeds budget -> skip
    assert size_position(500000, 125, 105, max_lots=2).lots == 2
    assert size_position(0, 125, 105).lots == 0


def test_split_tranches():
    assert split_tranches(3, [500, 1000]) == [(1, 500), (2, 1000)]
    assert split_tranches(1, [500, 1000]) == [(1, 500)]
    assert split_tranches(4, [140]) == [(4, 140)]
    assert split_tranches(0, [140]) == []


def test_round_to_tick_and_market_hours():
    assert round_to_tick(104.99999999999, 0.05, "down") == 105.0  # float noise
    assert round_to_tick(105.07, 0.05, "down") == 105.05
    assert round_to_tick(30.03, 0.05) == 30.05
    assert is_market_open(datetime(2026, 9, 23, 10, 0, tzinfo=IST))
    assert not is_market_open(datetime(2026, 9, 23, 15, 28, tzinfo=IST))
    assert not is_market_open(datetime(2026, 9, 26, 10, 0, tzinfo=IST))  # Saturday
    assert not is_market_open(datetime(2026, 9, 23, 10, 0, tzinfo=IST), {date(2026, 9, 23)})


# ---------------------------------------------------------------------------
# Executor (against a fake broker)
# ---------------------------------------------------------------------------

class FakeBroker(FnOBroker):
    name = "fake"

    def __init__(self, protects_on_entry=True, ltp=100.0, fill="full", fill_price=99.4, raw_status=""):
        super().__init__(dry_run=False)
        self.protects_on_entry = protects_on_entry
        self._ltp = ltp
        self.fill = fill  # full | none | partial
        self.fill_price = fill_price  # executed price the "broker" reports (None = not reported)
        self.raw_status = raw_status  # e.g. "CLOSED": a Dhan leg already executed
        self.modify_fails = 0
        self.entries, self.protections, self.cancelled, self.status_kinds = [], [], [], []
        self.modified = []

    def connect(self):
        pass

    def ltp(self, contract):
        return self._ltp

    def place_entry(self, contract, qty, price, stop_loss, target):
        oid = f"E{len(self.entries) + 1}"
        self.entries.append(dict(id=oid, qty=qty, price=price, sl=stop_loss, target=target))
        return oid

    def entry_status(self, order_id, protected=True):
        e = next(e for e in self.entries if e["id"] == order_id)
        self.status_kinds.append(protected)
        if self.fill == "full":
            return OrderState(COMPLETE, e["qty"], self.fill_price, self.raw_status)
        if self.fill == "partial":
            filled = e["qty"] // 3  # 1 of 3 lots
            return OrderState(CANCELLED if order_id in self.cancelled else OPEN, filled, self.fill_price)
        return OrderState(CANCELLED if order_id in self.cancelled else OPEN, 0)

    def cancel_entry(self, order_id, protected=True):
        self.cancelled.append(order_id)

    def place_protection(self, contract, qty, stop_loss, target, last_price):
        self.protections.append(dict(qty=qty, sl=stop_loss, target=target, last_price=last_price))
        return f"G{len(self.protections)}"

    def modify_protection(self, order_id, stop_loss, target):
        if self.modify_fails:
            self.modify_fails -= 1
            raise RuntimeError("Dhan said no")
        self.modified.append((order_id, stop_loss, target))


async def _no_sleep(_):
    await asyncio.sleep(0)


def _executor(tmp_path, broker, resolver, db="j.db", **overrides):
    settings = Settings(**{"capital_per_trade": 50000, "dry_run": False, "entry_timeout_sec": 300,
                           "journal_path": str(tmp_path / db), **overrides})
    journal = J.Journal(settings.journal_path)
    clock = lambda: datetime(2026, 9, 23, 10, 0, tzinfo=IST)  # noqa: E731
    return FnOExecutor(settings, broker, resolver, journal, Notifier(), clock=clock, sleep=_no_sleep), journal


def _payload(text=POLYCAB, sid="-100:1", label="calls"):
    d = parse_fno_regex(text).model_dump()
    d.update(asset_class="FNO", signal_id=sid, channel_label=label, source="REGEX")
    return d


@pytest.mark.asyncio
async def test_executor_dhan_style_places_one_super_order_at_percent_levels(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=True, ltp=100.0)
    ex, journal = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.PROTECTED
    # Super order at limit 100.1 (legs only act once the entry executes): SL 97.05 / T 103.15 ...
    # The signal's own SL 30 / targets 500,1000 are ignored.
    assert [(e["qty"], e["price"], e["sl"], e["target"]) for e in broker.entries] == [(375, 107.1,103.85, 110.35)]
    # ... then moved to ±3% of the EXECUTED price Dhan reports (99.4): SL 96.40 / T 102.40
    assert broker.modified == [("E1", 96.4, 102.4)]
    assert journal.get_signal("-100:1")["avg_price"] == 99.4
    assert journal.get_signal("-100:1")["filled_qty"] == 375


@pytest.mark.asyncio
async def test_executor_kite_style_entry_then_oco(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=False, ltp=100.0)
    ex, _ = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.PROTECTED
    assert [(e["qty"], e["target"]) for e in broker.entries] == [(375, None)]
    # GTT placed only after the fill, at ±3% of the executed price 99.4 (not the limit 100.1)
    assert [(p["qty"], p["sl"], p["target"]) for p in broker.protections] == [(375, 96.4, 102.4)]


@pytest.mark.asyncio
async def test_executor_partial_fill_timeout_protects_filled_only(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=False, fill="partial")
    ex, journal = _executor(tmp_path, broker, resolver, entry_timeout_sec=0)
    assert await ex.handle(_payload()) == J.PROTECTED
    assert broker.cancelled == ["E1"]
    assert [(p["qty"], p["target"]) for p in broker.protections] == [(125, 102.4)]
    assert journal.get_signal("-100:1")["filled_qty"] == 125


@pytest.mark.asyncio
async def test_executor_no_fill_cancels(tmp_path, resolver):
    broker = FakeBroker(fill="none")
    ex, _ = _executor(tmp_path, broker, resolver, entry_timeout_sec=0)
    assert await ex.handle(_payload()) == J.CANCELLED
    assert broker.cancelled == ["E1"]


@pytest.mark.asyncio
async def test_executor_skips_chase_budget_and_dedups(tmp_path, resolver):
    ex, _ = _executor(tmp_path, FakeBroker(ltp=110.0), resolver)  # > 105 * 1.03
    assert await ex.handle(_payload(sid="a")) == J.REJECTED
    assert await ex.handle(_payload(sid="a")) == "DUPLICATE"


    small = FakeBroker()
    ex, _ = _executor(tmp_path, small, resolver, db="c.db", capital_per_trade=10000)
    assert await ex.handle(_payload(sid="c")) == J.REJECTED
    assert small.entries == []


@pytest.mark.asyncio
async def test_executor_market_closed_and_channel_filter(tmp_path, resolver):
    broker = FakeBroker()
    ex, _ = _executor(tmp_path, broker, resolver, allowed_channels=["calls"])
    ex.clock = lambda: datetime(2026, 9, 23, 16, 0, tzinfo=IST)
    assert await ex.handle(_payload(sid="x")) == J.REJECTED
    assert await ex.handle(_payload(sid="y", label="other")) == "IGNORED"
    assert broker.entries == []


@pytest.mark.asyncio
async def test_reconcile_resumes_protection_after_crash(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=False)
    ex, journal = _executor(tmp_path, broker, resolver)
    # Simulate a crash right after the entry was placed and journaled
    sid = "-100:9"
    journal.try_claim(sid, _payload(sid=sid))
    contract = resolver.resolve("POLYCAB", 8000, "PE", "OCT", today=TODAY)
    journal.update_signal(sid, J.ENTRY_PLACED, contract=json.dumps(contract.to_dict()),
                          entry_deadline=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
    oid = broker.place_entry(contract, 375, 105, 30, None)
    journal.add_order(sid, J.ENTRY, oid, 375, 105, 30, None, "OPEN")

    await ex.reconcile()
    assert journal.get_signal(sid)["status"] == J.PROTECTED
    # Levels from the executed price the broker reports (99.4), not the limit 105
    assert [(p["qty"], p["sl"], p["target"]) for p in broker.protections] == [(375, 96.4, 102.4)]


# ---------------------------------------------------------------------------
# Broadcaster routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_main_broadcasts_fresh_fno_and_skips_stale(monkeypatch):
    import main

    sent = []

    async def fake_broadcast(data):
        sent.append(data)

    async def fake_log(_):
        return None

    monkeypatch.setattr(main.manager, "broadcast", fake_broadcast)
    monkeypatch.setattr(main, "send_trade_log", fake_log)
    mapping = SimpleNamespace(label="calls")

    fresh = SimpleNamespace(date=datetime.now(timezone.utc))
    assert await main.handle_fno_message(POLYCAB, fresh, "-100:5", mapping) is True
    assert sent[0]["asset_class"] == "FNO" and sent[0]["signal_id"] == "-100:5"
    assert sent[0]["symbol"] == "POLYCAB" and sent[0]["channel_label"] == "calls"

    stale = SimpleNamespace(date=datetime.now(timezone.utc) - timedelta(hours=1))
    assert await main.handle_fno_message(LT, stale, "-100:6", mapping) is True
    assert len(sent) == 1

    plain = SimpleNamespace(date=datetime.now(timezone.utc))
    assert await main.handle_fno_message("#RELIANCE buy @2900", plain, "-100:7", mapping) is False


# ---------------------------------------------------------------------------
# process_telegram_message: FnO routing must not change or slow the equity path
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(monkeypatch):
    """main.process_telegram_message with all I/O stubbed; records what each path saw."""
    import main
    from config import ChannelMapping

    rec = SimpleNamespace(broadcasts=[], equity_calls=[], fno_calls=[])

    async def noop(*a, **k):
        return None

    async def fake_broadcast(data):
        rec.broadcasts.append(data)

    async def fake_equity(text, timestamp=""):
        rec.equity_calls.append(text)
        return {"stock_symbol": "RELIANCE", "entry_price": 2900.0, "order_type": "LIMIT",
                "source": "REGEX", "raw_text": text, "timestamp": timestamp}

    real_fno = main.hybrid_fno_extract

    async def spy_fno(text):
        rec.fno_calls.append(text)
        return await real_fno(text)

    for name in ("send_to_notion_async", "send_to_discord_async", "send_trade_log", "extract_trade_async"):
        monkeypatch.setattr(main, name, noop)
    for name in ("write_last_date", "write_checkpoint"):
        monkeypatch.setattr(main, name, lambda *a, **k: None)
    monkeypatch.setattr(main, "read_last_date", lambda *a: "2026-09-23")
    monkeypatch.setattr(main.manager, "broadcast", fake_broadcast)
    monkeypatch.setattr(main, "hybrid_extract_trade", fake_equity)
    monkeypatch.setattr(main, "hybrid_fno_extract", spy_fno)

    def run(text, enable_trading=True, enable_fno=True, msg_id=1):
        mapping = ChannelMapping(-100, "", "calls", enable_trading=enable_trading, enable_fno_trading=enable_fno)
        monkeypatch.setitem(main.CHANNEL_MAP, -100, mapping)
        msg = SimpleNamespace(id=msg_id, text=text, media=None, date=datetime.now(timezone.utc))
        asyncio.run(main.process_telegram_message(msg, -100))
        return rec

    return run


EQUITY_MSGS = ["#RELIANCE @2900", "Buy #SBIN at 810 SL 780", "Bought TATAMOTORS cmp 950"]


@pytest.mark.parametrize("enable_fno", [False, True])
def test_equity_messages_skip_fno_path_entirely(pipeline, enable_fno):
    for i, text in enumerate(EQUITY_MSGS):
        rec = pipeline(text, enable_fno=enable_fno, msg_id=i)
    assert rec.fno_calls == []  # no FnO parsing, no Gemini call, no await before equity
    assert rec.equity_calls == EQUITY_MSGS
    assert [b["asset_class"] for b in rec.broadcasts] == ["EQUITY"] * 3
    assert rec.broadcasts[0]["stock_symbol"] == "RELIANCE" and rec.broadcasts[0]["signal_id"] == "-100:0"


def test_option_message_goes_to_fno_only(pipeline):
    rec = pipeline(POLYCAB)
    assert rec.equity_calls == []
    assert [b["asset_class"] for b in rec.broadcasts] == ["FNO"]


def test_unparseable_option_message_never_falls_through_to_equity(pipeline, monkeypatch):
    import services.fno_parser as fp

    async def no_gemini(text):
        return None

    monkeypatch.setattr(fp, "parse_fno_gemini", no_gemini)
    rec = pipeline("#LT 3900PE looking strong")  # option-looking, no entry price: must not become an LT equity buy
    assert rec.equity_calls == [] and rec.broadcasts == []


def test_channel_without_fno_flag_behaves_exactly_as_before(pipeline):
    rec = pipeline(POLYCAB, enable_fno=False)
    assert rec.fno_calls == []
    assert rec.broadcasts == []  # equity filter blocks it, as it always did


# ---------------------------------------------------------------------------
# FNO_SL_TARGET_ENABLED=false: plain entry, no SL/targets at the broker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("protects_on_entry", [True, False])  # Dhan-style and Kite-style brokers
async def test_sl_target_disabled_places_one_plain_entry(tmp_path, resolver, protects_on_entry):
    broker = FakeBroker(protects_on_entry=protects_on_entry, ltp=100.0)
    ex, journal = _executor(tmp_path, broker, resolver, sl_target_enabled=False)
    assert await ex.handle(_payload()) == J.ENTERED
    assert [(e["qty"], e["sl"], e["target"]) for e in broker.entries] == [(375, None, None)]
    assert broker.protections == []  # no GTT OCO either
    assert set(broker.status_kinds) == {False}  # polled as a plain order, not a super order
    assert journal.get_signal("-100:1")["status"] == J.ENTERED


@pytest.mark.asyncio
async def test_sl_target_enabled_is_the_default(tmp_path, resolver):
    assert Settings().sl_target_enabled is True
    broker = FakeBroker(protects_on_entry=True)
    ex, _ = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.PROTECTED
    assert [(e["sl"], e["target"]) for e in broker.entries] == [(103.85, 110.35)]
    assert broker.modified == [("E1", 96.4, 102.4)]
    assert set(broker.status_kinds) == {True}


@pytest.mark.asyncio
async def test_reconcile_keeps_a_disabled_trade_unprotected(tmp_path, resolver):
    """A trade entered with SL/target off must not get protection after a restart with it on."""
    broker = FakeBroker(protects_on_entry=False)
    ex, journal = _executor(tmp_path, broker, resolver)  # flag ON now
    sid = "-100:10"
    journal.try_claim(sid, _payload(sid=sid))
    contract = resolver.resolve("POLYCAB", 8000, "PE", "OCT", today=TODAY)
    journal.update_signal(sid, J.ENTRY_PLACED, contract=json.dumps(contract.to_dict()),
                          entry_deadline=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
    oid = broker.place_entry(contract, 375, 105, None, None)
    journal.add_order(sid, J.ENTRY, oid, 375, 105, None, None, "OPEN")  # entered while flag was OFF
    await ex.reconcile()
    assert journal.get_signal(sid)["status"] == J.ENTERED and broker.protections == []


def test_dhan_plain_vs_super_order_payloads(monkeypatch):
    from fno_agent.brokers.dhan import DhanFnOBroker
    from fno_agent.instruments import Contract

    calls = []
    broker = DhanFnOBroker("1", access_token="T", dry_run=False)
    monkeypatch.setattr(broker, "_request", lambda m, path, body=None: calls.append((m, path, body)) or
                        {"orderId": "9", "orderStatus": "PENDING", "filledQty": 125})
    c = Contract("POLYCAB", 8000, "PE", date(2026, 10, 27), "NFO", 125, 0.05, security_id="104253",
                 exchange_segment="NSE_FNO")

    broker.place_entry(c, 125, 105, 30, 500)
    broker.place_entry(c, 125, 105, None, None)
    (m1, p1, b1), (m2, p2, b2) = calls
    assert p1 == "/super/orders" and b1["stopLossPrice"] == 30 and b1["targetPrice"] == 500
    assert p2 == "/orders" and "stopLossPrice" not in b2 and b2["productType"] == "MARGIN"

    calls.clear()
    broker.entry_status("9", protected=False)
    broker.cancel_entry("9", protected=False)
    assert [(m, p) for m, p, _ in calls] == [("GET", "/orders/9"), ("DELETE", "/orders/9")]


# ---------------------------------------------------------------------------
# FNO_SL_PCT / FNO_TARGET_PCT (default 3% each, signal SL/targets ignored)
# ---------------------------------------------------------------------------

def test_default_sl_target_pct_is_3():
    s = Settings()
    assert (s.sl_pct, s.target_pct) == (3.0, 3.0)


@pytest.mark.asyncio
async def test_custom_percentages_and_signal_without_sl_target(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=True, ltp=100.0)
    ex, _ = _executor(tmp_path, broker, resolver, sl_pct=10, target_pct=20)
    assert await ex.handle(_payload(text="#POLYCAB 8000 PE OCT @90-105")) == J.PROTECTED
    # executed 99.4: SL -10% = 89.46 -> 89.45; T +20% = 119.28 -> 119.30
    assert broker.modified == [("E1", 89.45, 119.3)]


def test_levels_stay_strictly_around_cheap_premiums():
    from fno_agent.executor import levels
    from fno_agent.instruments import Contract

    c = Contract("X", 1, "CE", date(2026, 10, 27), "NFO", 125, 0.05)
    assert levels(0.50, c, 3, 3) == (0.45, [0.55])  # 3% of 0.50 is < 1 tick: still one tick away
    assert levels(1.00, c, 3, 3) == (0.95, [1.05])


@pytest.mark.asyncio
async def test_premium_at_minimum_tick_is_skipped(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=True, ltp=0.05)
    ex, _ = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload(text="#POLYCAB 8000 PE OCT @0.05")) == J.REJECTED
    assert broker.entries == []


def test_broadcaster_sends_option_signal_without_sl_target(pipeline):
    rec = pipeline("#LT 3900PE @80")
    assert [b["asset_class"] for b in rec.broadcasts] == ["FNO"] and rec.equity_calls == []
    assert (rec.broadcasts[0]["stop_loss"], rec.broadcasts[0]["targets"]) == (None, [])



# ---------------------------------------------------------------------------
# SL/target from the EXECUTED price only, and only after execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("protects_on_entry", [True, False])
async def test_nothing_is_placed_or_moved_when_the_entry_does_not_execute(tmp_path, resolver, protects_on_entry):
    broker = FakeBroker(protects_on_entry=protects_on_entry, fill="none")
    ex, _ = _executor(tmp_path, broker, resolver, entry_timeout_sec=0)
    assert await ex.handle(_payload()) == J.CANCELLED
    assert broker.protections == [] and broker.modified == []


@pytest.mark.asyncio
async def test_kite_without_broker_reported_price_places_nothing_and_alerts(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=False, fill_price=None)
    ex, journal = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.UNPROTECTED
    assert broker.protections == []  # never guessed from the limit price


@pytest.mark.asyncio
async def test_dhan_without_broker_reported_price_keeps_legs_and_warns(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=True, fill_price=None)
    ex, journal = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.PROTECTED
    assert broker.modified == []
    assert "executed price unavailable" in journal.get_signal("-100:1")["reason"]


@pytest.mark.asyncio
async def test_dhan_leg_move_is_retried_then_reported(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=True)
    broker.modify_fails = 2  # third attempt succeeds
    ex, _ = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.PROTECTED
    assert broker.modified == [("E1", 96.4, 102.4)]

    broker = FakeBroker(protects_on_entry=True)
    broker.modify_fails = 5  # never succeeds
    ex, journal = _executor(tmp_path, broker, resolver, db="fail.db")
    assert await ex.handle(_payload(sid="z")) == J.PROTECTED
    assert "not moved to executed price" in journal.get_signal("z")["reason"]


@pytest.mark.asyncio
async def test_dhan_leg_already_executed_is_not_touched(tmp_path, resolver):
    broker = FakeBroker(protects_on_entry=True, raw_status="CLOSED")
    ex, _ = _executor(tmp_path, broker, resolver)
    assert await ex.handle(_payload()) == J.PROTECTED
    assert broker.modified == []


def test_dhan_modify_and_trades_fallback(monkeypatch):
    from fno_agent.brokers.dhan import DhanFnOBroker

    broker = DhanFnOBroker("1", access_token="T", dry_run=False)
    calls = []

    def fake_request(method, path, body=None):
        calls.append((method, path, body))
        if path.startswith("/super/orders") and method == "GET":
            return [{"orderId": "9", "orderStatus": "TRADED", "filledQty": 250, "averageTradedPrice": 0}]
        if path == "/trades/9":
            return [{"tradedPrice": 99.0, "tradedQuantity": 125}, {"tradedPrice": 100.0, "tradedQuantity": 125}]
        return {"orderId": "9", "orderStatus": "TRADED"}

    monkeypatch.setattr(broker, "_request", fake_request)
    assert broker.executed_price("9") == 99.5  # weighted average of the trades
    calls.clear()
    broker.modify_protection("9", 96.5, 102.5)
    assert [(m, p, b["legName"]) for m, p, b in calls] == [
        ("PUT", "/super/orders/9", "TARGET_LEG"), ("PUT", "/super/orders/9", "STOP_LOSS_LEG")]
    assert calls[0][2]["targetPrice"] == 102.5 and calls[1][2]["stopLossPrice"] == 96.5
