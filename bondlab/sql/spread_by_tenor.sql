-- Average bid-offer width (bp of yield) by sector and remaining-maturity bucket.
-- One-sided quotes are excluded; each quote is counted once, however many
-- polls saw it.
WITH distinct_quotes AS (
    SELECT DISTINCT code, quote_time, bid_yield, offer_yield
    FROM quotes_wide
    WHERE bid_yield IS NOT NULL AND offer_yield IS NOT NULL
),
q AS (
    SELECT s.sector,
           (julianday(s.maturity) - julianday(d.quote_time)) / 365.25 AS tenor,
           (d.bid_yield - d.offer_yield) * 100.0               AS spread_bp
    FROM distinct_quotes AS d
    JOIN securities AS s USING (code)
)
SELECT sector,
       CASE WHEN tenor < 2  THEN '0-2y'
            WHEN tenor < 5  THEN '2-5y'
            WHEN tenor < 10 THEN '5-10y'
            ELSE '10y+' END             AS bucket,
       COUNT(*)                          AS n_quotes,
       ROUND(AVG(spread_bp), 2)          AS avg_spread_bp,
       ROUND(MIN(spread_bp), 2)          AS min_spread_bp,
       ROUND(MAX(spread_bp), 2)          AS max_spread_bp
FROM q
GROUP BY sector, bucket
ORDER BY sector, MIN(tenor);
