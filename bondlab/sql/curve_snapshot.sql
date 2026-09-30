-- The curve-fitting input at time :at: for each bond in :sector, the latest
-- two-sided quote polled at or before :at and no older than :max_age_s seconds.
-- Only the polls inside the age window are pivoted, so the query touches a few
-- thousand rows of quotes_long however long the log grows (the primary key
-- starts with poll_time).
WITH recent AS (
    SELECT poll_time,
           code,
           datetime(MAX(CASE WHEN field = 'quote_time' THEN value END), 'unixepoch') AS quote_time,
           MAX(CASE WHEN field = 'mid_yield' THEN value END) AS mid_yield
    FROM quotes_long
    WHERE poll_time BETWEEN datetime(:at, '-' || CAST(:max_age_s AS INTEGER) || ' seconds') AND :at
      AND field IN ('quote_time', 'mid_yield')
    GROUP BY poll_time, code
),
ranked AS (
    SELECT r.code, r.poll_time, r.quote_time, r.mid_yield,
           s.coupon, s.freq, s.issue, s.maturity,
           ROW_NUMBER() OVER (PARTITION BY r.code ORDER BY r.poll_time DESC) AS rn
    FROM recent AS r
    JOIN securities AS s USING (code)
    WHERE s.sector = :sector
      AND r.mid_yield IS NOT NULL
      AND CAST(strftime('%s', :at) AS INTEGER) - CAST(strftime('%s', r.quote_time) AS INTEGER) <= :max_age_s
)
SELECT code, poll_time, quote_time, mid_yield, coupon, freq, issue, maturity
FROM ranked
WHERE rn = 1
ORDER BY maturity;
