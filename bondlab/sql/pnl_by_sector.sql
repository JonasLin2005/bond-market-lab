-- P&L attribution summed by sector over the whole sheet, in thousands.
SELECT s.sector,
       COUNT(DISTINCT p.code)          AS positions,
       ROUND(SUM(p.carry_k), 1)        AS carry,
       ROUND(SUM(p.rolldown_k), 1)     AS rolldown,
       ROUND(SUM(p.level_k), 1)        AS level,
       ROUND(SUM(p.slope_k), 1)        AS slope,
       ROUND(SUM(p.curvature_k), 1)    AS curvature,
       ROUND(SUM(p.spread_k), 1)       AS spread,
       ROUND(SUM(p.total_k), 1)        AS total,
       ROUND(100.0 * AVG(p.stale), 1)  AS pct_stale_marks
FROM pnl_daily AS p
JOIN securities AS s USING (code)
GROUP BY s.sector
ORDER BY total DESC;
