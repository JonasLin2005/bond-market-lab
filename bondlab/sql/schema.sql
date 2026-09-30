-- Reference data for the book.
CREATE TABLE IF NOT EXISTS securities (
    code      TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    sector    TEXT NOT NULL CHECK (sector IN
                  ('treasury', 'policy_bank', 'local_gov', 'urban_invest', 'bank_capital')),
    issuer    TEXT NOT NULL,
    coupon    REAL NOT NULL,          -- percent
    freq      INTEGER NOT NULL,
    issue     TEXT NOT NULL,          -- YYYY-MM-DD
    maturity  TEXT NOT NULL
);

-- Every polled snapshot, one row per field. Long format keeps the table stable
-- when the feed adds or drops a field; quotes_wide pivots it back.
CREATE TABLE IF NOT EXISTS quotes_long (
    poll_time TEXT NOT NULL,          -- YYYY-MM-DD HH:MM:SS
    code      TEXT NOT NULL REFERENCES securities(code),
    field     TEXT NOT NULL,
    value     REAL,                   -- times are stored as unix seconds
    PRIMARY KEY (poll_time, code, field)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS quotes_long_code_time ON quotes_long (code, poll_time);

CREATE TABLE IF NOT EXISTS bars_daily (
    code TEXT NOT NULL REFERENCES securities(code),
    time TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    open_yield REAL, high_yield REAL, low_yield REAL, close_yield REAL, close_mid_yield REAL,
    volume REAL, n_trades INTEGER,
    PRIMARY KEY (code, time)
);

CREATE TABLE IF NOT EXISTS bars_minute (
    code TEXT NOT NULL REFERENCES securities(code),
    time TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    open_yield REAL, high_yield REAL, low_yield REAL, close_yield REAL, close_mid_yield REAL,
    volume REAL, n_trades INTEGER,
    PRIMARY KEY (code, time)
);

CREATE VIEW IF NOT EXISTS quotes_wide AS
SELECT
    poll_time,
    code,
    datetime(MAX(CASE WHEN field = 'quote_time'      THEN value END), 'unixepoch') AS quote_time,
    MAX(CASE WHEN field = 'bid_yield'   THEN value END) AS bid_yield,
    MAX(CASE WHEN field = 'offer_yield' THEN value END) AS offer_yield,
    MAX(CASE WHEN field = 'mid_yield'   THEN value END) AS mid_yield,
    MAX(CASE WHEN field = 'bid_clean'   THEN value END) AS bid_clean,
    MAX(CASE WHEN field = 'offer_clean' THEN value END) AS offer_clean,
    MAX(CASE WHEN field = 'bid_size'    THEN value END) AS bid_size,
    MAX(CASE WHEN field = 'offer_size'  THEN value END) AS offer_size,
    datetime(MAX(CASE WHEN field = 'last_trade_time' THEN value END), 'unixepoch') AS last_trade_time,
    MAX(CASE WHEN field = 'last_yield'  THEN value END) AS last_yield,
    MAX(CASE WHEN field = 'last_clean'  THEN value END) AS last_clean,
    MAX(CASE WHEN field = 'last_volume' THEN value END) AS last_volume,
    MAX(CASE WHEN field = 'accrued'     THEN value END) AS accrued
FROM quotes_long
GROUP BY poll_time, code;

-- The daily position and P&L sheet (thousands; face in millions).
CREATE TABLE IF NOT EXISTS pnl_daily (
    date TEXT NOT NULL,
    code TEXT NOT NULL REFERENCES securities(code),
    face_mm REAL, clean REAL, yield_pct REAL, z_spread_bp REAL, stale INTEGER,
    dv01_k REAL, carry_k REAL, rolldown_k REAL, level_k REAL, slope_k REAL,
    curvature_k REAL, spread_k REAL, unexplained_k REAL, total_k REAL,
    PRIMARY KEY (date, code)
);

-- How the simulated market behind this database was generated, so any command
-- can rebuild it and score results against the truth.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
