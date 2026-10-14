"""
FnO signal execution: validate → resolve contract → size → LIMIT entry → wait for fill → broker-side SL/target.

SL/target are a fixed % of the EXECUTED price reported by the broker: FNO_SL_PCT / FNO_TARGET_PCT
(default 3% each). The signal's own SL/targets are ignored. Nothing acts unless the entry executes:
- Dhan: super order; its legs only act once the entry executes (and are cancelled with it). After the
  fill, both legs are moved to ±X% of Dhan's averageTradedPrice.
- Kite: plain entry; after the fill, a GTT OCO at ±X% of Kite's average_price.

Every step is written to the journal so a crash mid-trade can be resumed by reconcile().
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from fno_agent import journal as J
from fno_agent.brokers.base import TERMINAL, BrokerAuthError, BrokerError, FnOBroker, OrderState
from fno_agent.instruments import Contract, ContractNotFound, InstrumentResolver
from fno_agent.market import is_market_open, now_ist, round_to_tick
from fno_agent.notifier import Notifier
from fno_agent.settings import Settings
from fno_agent.sizer import size_position, split_tranches
from services.fno_parser import FnOSignal, _validate

logger = logging.getLogger(__name__)

ENTRY_TICKS_ABOVE_LTP = 2  # limit = min(entry_max, LTP + 2 ticks)


def describe(sig: FnOSignal) -> str:
    exp = f" {sig.expiry_day or ''}{sig.expiry_month}" if sig.expiry_month else ""
    return f"{sig.symbol} {sig.strike:g}{sig.option_type}{exp} @ {sig.entry_min:g}-{sig.entry_max:g}"


def levels(entry: float, contract: Contract, sl_pct: float, target_pct: float) -> Tuple[float, List[float]]:
    """
    Stop-loss and a single target as % of the entry price, on the contract's tick grid.
    SL rounds down and target rounds up, so both stay strictly away from the entry even for cheap
    premiums (e.g. entry 0.50 at 3% -> SL 0.45, target 0.55).
    """
    tick = contract.tick_size
    sl = round_to_tick(entry * (1 - sl_pct / 100), tick, "down")
    sl = sl if sl < entry else round_to_tick(entry - tick, tick, "down")
    target = round_to_tick(entry * (1 + target_pct / 100), tick, "up")
    target = target if target > entry else round_to_tick(entry + tick, tick, "up")
    return max(sl, tick), [target]


class FnOExecutor:
    def __init__(
        self,
        settings: Settings,
        broker: FnOBroker,
        resolver: InstrumentResolver,
        journal: J.Journal,
        notifier: Notifier,
        holidays: Iterable = (),
        clock: Callable[[], datetime] = now_ist,
        sleep: Callable = asyncio.sleep,
    ):
        self.s = settings
        self.broker = broker
        self.resolver = resolver
        self.journal = journal
        self.notifier = notifier
        self.holidays = set(holidays)
        self.clock = clock
        self.sleep = sleep

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def handle(self, signal: dict) -> str:
        """Process one broadcast FnO signal. Returns the final journal status (or IGNORED/DUPLICATE)."""
        sid = signal.get("signal_id") or f"nosid:{abs(hash(signal.get('raw_text', '')))}"
        label = str(signal.get("channel_label", "")).lower()
        if self.s.allowed_channels and label not in self.s.allowed_channels:
            logger.info(f"Ignoring {sid}: channel '{label}' not in ALLOWED_CHANNELS")
            return "IGNORED"
        if not self.journal.try_claim(sid, signal):
            logger.info(f"Duplicate signal {sid} ignored.")
            return "DUPLICATE"

        try:
            return await self._execute(sid, signal)
        except ContractNotFound as e:
            return await self._reject(sid, f"contract not found: {e}")
        except BrokerAuthError as e:
            return await self._fail(sid, f"BROKER AUTH FAILED: {e}")
        except Exception as e:
            logger.exception(f"Signal {sid} failed")
            return await self._fail(sid, f"error: {e}")

    async def _reject(self, sid: str, reason: str) -> str:
        self.journal.update_signal(sid, J.REJECTED, reason)
        await self.notifier.send(f"⛔ Skipped {sid}: {reason}")
        return J.REJECTED

    async def _fail(self, sid: str, reason: str) -> str:
        row = self.journal.get_signal(sid)
        if row and row["status"] in J.PENDING_STATES:
            # Orders may be live at the broker: keep it pending so reconcile() picks it up.
            self.journal.update_signal(sid, reason=reason)
            await self.notifier.send(f"🚨 {sid} needs attention (orders may be live): {reason}")
            return row["status"]
        self.journal.update_signal(sid, J.ERROR, reason)
        await self.notifier.send(f"🚨 {sid} failed: {reason}")
        return J.ERROR

    # ------------------------------------------------------------------
    # Validate → resolve → size → enter
    # ------------------------------------------------------------------

    async def _execute(self, sid: str, signal: dict) -> str:
        try:
            sig = _validate(FnOSignal.model_validate(signal))
        except Exception as e:
            sig = None
            logger.warning(f"Bad FnO payload {sid}: {e}")
        if sig is None:
            return await self._reject(sid, "invalid signal payload")
        desc = describe(sig)
        logger.info(f"Signal {sid}: {desc}")

        now = self.clock()
        if self.s.enforce_market_hours and not self.s.dry_run and not is_market_open(now, self.holidays):
            return await self._reject(sid, f"market closed ({now:%a %H:%M} IST): {desc}")

    contract = self.resolver.resolve(sig.symbol, sig.strike, sig.option_type,
                                         sig.expiry_month, sig.expiry_day, today=now.date())
        size = size_position(self.s.capital_per_trade, contract.lot_size, sig.entry_max * 1.03, self.s.max_lots)
        if size.lots == 0:
            return await self._reject(
                sid, f"1 lot of {contract.describe()} costs ₹{size.cost_per_lot:,.0f} "
                     f"> budget ₹{self.s.capital_per_trade:,.0f}")

        tick = contract.tick_size
        ltp = await asyncio.to_thread(self.broker.ltp, contract)
        if ltp is not None:
            if ltp > sig.entry_max * (1 + self.s.chase_pct / 100):
                return await self._reject(sid, f"missed entry: LTP {ltp} > {sig.entry_max} +{self.s.chase_pct}% ({desc})")
            
            # Limit price set to 3% above entry_max
            limit = sig.entry_max * 1.03
        else:
            logger.warning(f"No LTP for {contract.describe()}; using entry_max + 3% as limit.")
            
            # Fallback limit price set to 3% above entry_max
            limit = sig.entry_max * 1.03
            
        limit = round_to_tick(limit, tick, "down")
        sl, targets = levels(limit, contract, self.s.sl_pct, self.s.target_pct)
        if self.s.sl_target_enabled and sl >= limit:
            return await self._reject(sid, f"premium {limit} too small for a stop-loss below it ({desc})")
        protect = self.s.sl_target_enabled
        # The entry row's stop_loss records whether this trade gets SL/targets (None = disabled),
        # so a restart (reconcile) finishes it the same way.
        entry_sl = sl if protect else None

        deadline = datetime.now(timezone.utc) + timedelta(seconds=self.s.entry_timeout_sec)
        self.journal.update_signal(sid, contract=json.dumps(contract.to_dict()), planned_qty=size.qty,
                                   entry_deadline=deadline.isoformat())

        # Dhan: one super order per target tranche (SL/target attached). Kite: one entry, GTTs after fill.
        # SL/target disabled: one plain entry for the whole quantity.
        if protect and self.broker.protects_on_entry:
            plan = [(lots * contract.lot_size, tgt) for lots, tgt in split_tranches(size.lots, targets)]
        else:
            plan = [(size.qty, None)]

        placed, errors = 0, []
        for qty, tgt in plan:
            try:
                oid = await asyncio.to_thread(self.broker.place_entry, contract, qty, limit, entry_sl, tgt)
            except BrokerAuthError:
                if placed:
                    errors.append("auth failure mid-placement")
                    break
                raise
            except Exception as e:
                errors.append(str(e))
                continue
            self.journal.add_order(sid, J.ENTRY, oid, qty, limit, entry_sl, tgt, "OPEN")
            if not placed:
                self.journal.update_signal(sid, J.ENTRY_PLACED)
            placed += 1

        if not placed:
            return await self._reject(sid, f"entry rejected by broker: {'; '.join(errors)}")
        await self.notifier.send(
            f"🟢 BUY {size.lots} lot(s) = {size.qty} x {contract.describe()} LIMIT {limit} "
            f"(LTP {ltp if ltp is not None else 'n/a'}) "
            + (f"SL {sl} (-{self.s.sl_pct:g}%) T {targets[0]} (+{self.s.target_pct:g}%)" if protect
               else "NO SL/target (FNO_SL_TARGET_ENABLED=false)")
            + f" [{sid}]"
            + (f"\n⚠️ some tranches failed: {'; '.join(errors)}" if errors else "")
            + ("\nℹ️ signal says go slow/limited qty" if sig.caution else "")
        )
        return await self.await_fill_and_protect(sid)

    # ------------------------------------------------------------------
    # Fill watch → protection (also used by reconcile after a restart)
    # ------------------------------------------------------------------

    async def await_fill_and_protect(self, sid: str) -> str:
        row = self.journal.get_signal(sid)
        contract = Contract.from_dict(json.loads(row["contract"]))
        sig = FnOSignal.model_validate(json.loads(row["payload"]))
        deadline = datetime.fromisoformat(row["entry_deadline"])
        entries = self.journal.orders_for(sid, J.ENTRY)

        states: Dict[int, OrderState] = await self._poll_until(entries, deadline)
        filled = sum(st.filled_qty for st in states.values())
        still_open = [e["broker_order_id"] for e in entries if states[e["id"]].status not in TERMINAL]

        if still_open:
            await self.notifier.send(f"🚨 {sid}: could not cancel entry order(s) {still_open}; check broker!")

        if filled == 0:
            self.journal.update_signal(sid, J.CANCELLED, "no fill before timeout", filled_qty=0)
            await self.notifier.send(f"⌛ No fill for {contract.describe()} within "
                                     f"{self.s.entry_timeout_sec}s; entry cancelled [{sid}]")
            return J.CANCELLED

        avg = await self._executed_price(entries, states)
        self.journal.update_signal(sid, J.FILLED, filled_qty=filled, avg_price=avg)

        if entries[0]["stop_loss"] is None:  # SL/target disabled when this trade was entered
            self.journal.update_signal(sid, J.ENTERED, "SL/target disabled")
            await self.notifier.send(f"✅ Filled {filled} x {contract.describe()} avg {avg or '?'}; "
                                     f"NO stop-loss/target placed (FNO_SL_TARGET_ENABLED=false) [{sid}]")
            return J.ENTERED

        if self.broker.protects_on_entry:
            return await self._reanchor(sid, contract, entries, states, filled, avg)

        return await self._protect(sid, sig, contract, filled, avg)

    async def _executed_price(self, entries, states: Dict[int, OrderState]) -> Optional[float]:
        """Quantity-weighted average executed price, as reported by the broker (None if unknown)."""
        value, qty = 0.0, 0
        for e in entries:
            st = states[e["id"]]
            if not st.filled_qty:
                continue
            price = st.avg_price or await asyncio.to_thread(
                self.broker.executed_price, e["broker_order_id"], e["target"] is not None)
            if not price:
                return None  # never guess: part of the fill has no broker-reported price
            value += price * st.filled_qty
            qty += st.filled_qty
        return round(value / qty, 4) if qty else None

    async def _reanchor(self, sid: str, contract: Contract, entries, states: Dict[int, OrderState],
                        filled: int, avg: Optional[float]) -> str:
        """Dhan super order filled: move its SL/target legs to ±X% of the executed price."""
        limit = entries[0]["price"]
        if avg is None:
            self.journal.update_signal(sid, J.PROTECTED, "executed price unavailable; legs at limit-based levels")
            await self.notifier.send(
                f"⚠️ Filled {filled} x {contract.describe()}, but {self.broker.name} did not report the executed "
                f"price. SL/target legs stay at ±% of the limit {limit} (SL {entries[0]['stop_loss']}, "
                f"T {entries[0]['target']}). Check them in the app. [{sid}]")
            return J.PROTECTED

        sl, targets = levels(avg, contract, self.s.sl_pct, self.s.target_pct)
        errors = []
        for e in entries:
            st = states[e["id"]]
            if not st.filled_qty:
                continue
            if st.message == "CLOSED":  # a leg already executed before we could move it
                continue
            for attempt in range(3):
                try:
                    await asyncio.to_thread(self.broker.modify_protection, e["broker_order_id"], sl, targets[0])
                    self.journal.update_order_levels(e["id"], sl, targets[0])
                    break
                except Exception as ex:
                    if attempt == 2:
                        errors.append(f"{e['broker_order_id']}: {ex}")
                    else:
                        await self.sleep(1)

        if errors:
            self.journal.update_signal(sid, J.PROTECTED, f"legs not moved to executed price: {'; '.join(errors)}")
            await self.notifier.send(
                f"🚨 Filled {filled} x {contract.describe()} @ {avg}, but the SL/target legs could not be moved "
                f"to SL {sl} / T {targets[0]}. They are still at SL {entries[0]['stop_loss']} / "
                f"T {entries[0]['target']} (from the limit {limit}). Fix in the Dhan app! {errors} [{sid}]")
            return J.PROTECTED

        self.journal.update_signal(sid, J.PROTECTED)
        await self.notifier.send(
            f"✅ Filled {filled} x {contract.describe()} @ {avg} (executed, from {self.broker.name}); "
            f"SL {sl} (-{self.s.sl_pct:g}%) / T {targets[0]} (+{self.s.target_pct:g}%) set at {self.broker.name} [{sid}]")
        return J.PROTECTED

    async def _poll_until(self, entries, deadline: datetime) -> Dict[int, OrderState]:
        states: Dict[int, OrderState] = {e["id"]: OrderState("OPEN") for e in entries}
        cancelled = False
        while True:
            for e in entries:
                if states[e["id"]].status in TERMINAL:
                    continue
                try:
                    st = await asyncio.to_thread(self.broker.entry_status, e["broker_order_id"],
                                                 e["target"] is not None)
                except BrokerError as ex:
                    logger.warning(f"Status check failed for {e['broker_order_id']}: {ex}")
                    continue
                states[e["id"]] = st
                self.journal.update_order(e["id"], st.status, st.filled_qty)

            if all(st.status in TERMINAL for st in states.values()) or cancelled:
                return states

            if datetime.now(timezone.utc) >= deadline:
                for e in entries:
                    if states[e["id"]].status not in TERMINAL:
                        try:
                            await asyncio.to_thread(self.broker.cancel_entry, e["broker_order_id"],
                                                    e["target"] is not None)
                        except Exception as ex:
                            logger.error(f"Cancel failed for {e['broker_order_id']}: {ex}")
                cancelled = True  # one final status poll to pick up partial fills
                await self.sleep(1)
                continue

            await self.sleep(self.s.poll_interval_sec)

    async def _protect(self, sid: str, sig: FnOSignal, contract: Contract, filled: int,
                       avg: Optional[float]) -> str:
        """Kite: after the fill, a GTT OCO at ±X% of the executed price reported by the broker."""
        if avg is None:
            self.journal.update_signal(sid, J.UNPROTECTED, "executed price unavailable")
            await self.notifier.send(f"🚨 UNPROTECTED: filled {filled} x {contract.describe()} but "
                                     f"{self.broker.name} did not report the executed price, so no SL/target "
                                     f"was placed. Place them manually! [{sid}]")
            return J.UNPROTECTED
        sl, targets = levels(avg, contract, self.s.sl_pct, self.s.target_pct)
        tranches = split_tranches(filled // contract.lot_size, targets)
        done = self.journal.orders_for(sid, J.PROTECTION)  # already placed before a crash
        last_price = await asyncio.to_thread(self.broker.ltp, contract) or avg

        errors = []
        for lots, tgt in tranches[len(done):]:
            qty = lots * contract.lot_size
            try:
                pid = await asyncio.to_thread(self.broker.place_protection, contract, qty, sl, tgt, last_price)
                self.journal.add_order(sid, J.PROTECTION, pid, qty, None, sl, tgt, "ACTIVE")
            except Exception as e:
                errors.append(f"{qty}@T{tgt}: {e}")

        if errors:
            self.journal.update_signal(sid, J.UNPROTECTED, "; ".join(errors))
            await self.notifier.send(f"🚨 UNPROTECTED position {filled} x {contract.describe()} — "
                                     f"place SL {sl} manually! {errors} [{sid}]")
            return J.UNPROTECTED

        self.journal.update_signal(sid, J.PROTECTED)
        await self.notifier.send(
            f"✅ Filled {filled} x {contract.describe()} @ {avg} (executed, from {self.broker.name}); "
            f"OCO SL {sl} (-{self.s.sl_pct:g}%) / T {targets[0]} (+{self.s.target_pct:g}%) [{sid}]")
        return J.PROTECTED

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    async def reconcile(self):
        """Resume trades interrupted by a restart."""
        for row in self.journal.signals_with_status(J.RECEIVED):
            # Crashed before any entry was recorded; an order may still have reached the broker.
            self.journal.update_signal(row["signal_id"], J.ERROR, "interrupted before entry was recorded")
            await self.notifier.send(f"🚨 {row['signal_id']} was interrupted before entry was recorded; "
                                     f"verify no stray order at the broker.")
        for row in self.journal.pending_signals():
            sid = row["signal_id"]
            logger.info(f"Reconciling {sid} ({row['status']})")
            await self.notifier.send(f"♻️ Resuming {sid} after restart ({row['status']})")
            try:
                await self.await_fill_and_protect(sid)
            except Exception as e:
                logger.exception(f"Reconcile failed for {sid}")
                await self._fail(sid, f"reconcile error: {e}")
