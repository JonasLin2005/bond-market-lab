-- One-minute OHLC of the polled mid yield (percent) for one bond, or all bonds
-- when :code is NULL. Open and close come from window functions over each minute.
WITH m AS (
    SELECT code,
           poll_time,
           substr(poll_time, 1, 16) || ':00' AS minute,
           mid_yield
    FROM quotes_wide
    WHERE mid_yield IS NOT NULL
      AND (:code IS NULL OR code = :code)
),
r AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY code, minute ORDER BY poll_time)      AS rn_first,
           ROW_NUMBER() OVER (PARTITION BY code, minute ORDER BY poll_time DESC) AS rn_last
    FROM m
)
SELECT code,
       minute,
       MAX(CASE WHEN rn_first = 1 THEN mid_yield END) AS open,
       MAX(mid_yield)                                AS high,
       MIN(mid_yield)                                AS low,
       MAX(CASE WHEN rn_last = 1 THEN mid_yield END)  AS close,
       COUNT(*)                                      AS n_polls
FROM r
GROUP BY code, minute
ORDER BY code, minute;
