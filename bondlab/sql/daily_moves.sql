-- Day-on-day change in closing yield (bp) from the daily bars, per bond:
-- average absolute move, largest move and the session it happened.
WITH d AS (
    SELECT code, time,
           (close_yield - LAG(close_yield) OVER (PARTITION BY code ORDER BY time)) * 100.0 AS chg_bp
    FROM bars_daily
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY code ORDER BY ABS(chg_bp) DESC) AS rn
    FROM d WHERE chg_bp IS NOT NULL
)
SELECT r.code,
       (SELECT COUNT(*) FROM bars_daily b WHERE b.code = r.code) AS sessions,
       ROUND((SELECT AVG(ABS(chg_bp)) FROM d WHERE d.code = r.code), 2) AS avg_abs_move_bp,
       ROUND(r.chg_bp, 2) AS largest_move_bp,
       substr(r.time, 1, 10) AS on_date
FROM ranked AS r
WHERE r.rn = 1
ORDER BY r.code;
