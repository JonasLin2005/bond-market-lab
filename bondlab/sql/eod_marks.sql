-- End-of-day marks for every live bond on every session between :start and :end.
-- The mark is the closing mid yield of the last session the bond traded, on or
-- before that day; mark_date says which session that was, so a stale mark on
-- an illiquid bond is visible rather than silent.
WITH days AS (
    SELECT DISTINCT substr(time, 1, 10) AS d
    FROM bars_daily
    WHERE substr(time, 1, 10) BETWEEN :start AND :end
),
grid AS (
    SELECT days.d, s.code, s.sector
    FROM days
    CROSS JOIN securities AS s
    WHERE s.issue <= days.d AND s.maturity > days.d
),
last_bar AS (
    SELECT g.d, g.code, g.sector,
           (SELECT b.time FROM bars_daily AS b
             WHERE b.code = g.code AND b.time < date(g.d, '+1 day')
             ORDER BY b.time DESC LIMIT 1) AS bar_time
    FROM grid AS g
)
SELECT l.d                    AS date,
       l.code,
       l.sector,
       COALESCE(b.close_mid_yield, b.close_yield) AS close_yield,
       substr(l.bar_time, 1, 10) AS mark_date,
       substr(l.bar_time, 1, 10) <> l.d AS stale
FROM last_bar AS l
LEFT JOIN bars_daily AS b ON b.code = l.code AND b.time = l.bar_time
ORDER BY l.code, l.d;
