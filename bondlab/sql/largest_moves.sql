-- Largest poll-to-poll changes in mid yield (bp), using LAG over each bond's polls.
WITH d AS (
    SELECT code,
           poll_time,
           mid_yield,
           LAG(mid_yield) OVER (PARTITION BY code ORDER BY poll_time) AS prev_mid,
           LAG(poll_time) OVER (PARTITION BY code ORDER BY poll_time) AS prev_time
    FROM quotes_wide
    WHERE mid_yield IS NOT NULL
)
SELECT code,
       prev_time,
       poll_time,
       ROUND((mid_yield - prev_mid) * 100.0, 2) AS change_bp
FROM d
WHERE prev_mid IS NOT NULL
ORDER BY ABS(mid_yield - prev_mid) DESC
LIMIT :n;
