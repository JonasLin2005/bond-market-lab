import pandas as pd
import pytest

from bondlab import store
from bondlab.api import Limits, MockQuoteAPI, SimClock
from bondlab.client import QuoteClient
from bondlab.logger import QuoteLogger
from bondlab.market import Market


@pytest.fixture(scope="session")
def market():
    return Market(start="2024-07-01", days=60, seed=3)


@pytest.fixture(scope="session")
def sessions(market):
    return pd.bdate_range(market.start.normalize(), market.minutes[-1].normalize())


@pytest.fixture(scope="session")
def lab(market, sessions):
    """A database with full daily history and two logged sessions."""
    clock = SimClock(sessions[-2] + pd.Timedelta(hours=8))
    api = MockQuoteAPI(market, clock, Limits(), seed=11)
    client = QuoteClient(api, seed=12)
    con = store.connect()
    store.write_securities(con, market.bonds)
    codes = [b.code for b in market.bonds]
    store.write_bars(con, client.daily_bars(codes, market.start, clock.now()), "bars_daily")
    logger = QuoteLogger(client, con)
    logger.run(codes, until=market.end, every_s=60)
    return {"con": con, "client": client, "clock": clock, "logger": logger, "codes": codes}


def fresh_api(market, when, **limits):
    clock = SimClock(when)
    return MockQuoteAPI(market, clock, Limits(**limits), seed=5), clock
