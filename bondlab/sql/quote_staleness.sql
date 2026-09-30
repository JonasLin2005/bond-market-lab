-- How old the live quote was when polled, per bond. A quote older than
-- :max_age_s seconds is stale: its yield reflects an earlier market.
WITH a AS (
    SELECT code,
           CAST(strftime('%s', poll_time) AS INTEGER)
             - CAST(strftime('%s', quote_time) AS INTEGER) AS age_s   -- exact seconds
    FROM quotes_wide
)
SELECT code,
       COUNT(*)                                           AS n_polls,
       ROUND(AVG(age_s) / 60.0, 2)                        AS avg_age_min,
       ROUND(MAX(age_s) / 60.0, 2)                        AS max_age_min,
       ROUND(100.0 * SUM(age_s > :max_age_s) / COUNT(*), 1) AS pct_stale
FROM a
GROUP BY code
ORDER BY pct_stale DESC, code;
