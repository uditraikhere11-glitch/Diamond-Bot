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
    # Super order at limit 108.15 (105 * 1.03 = 108.15); SL and T derived from limit
    # The signal's own SL 30 / targets 500,1000 are ignored.
    assert [(e["qty"], e["price"], e["sl"], e["target"]) for e in broker.entries] == [(375, 108.15, 104.9, 111.4)]
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
    # GTT placed only after the fill, at ±3% of the executed price 99.4 (not the limit 108.15)
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
    assert await ex.handle(_payload(sid="a")) == "DU
