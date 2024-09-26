-- SELECT B.Band_Name, C.Concert_Name, C.Date
-- FROM Bands B
-- JOIN Assignments A
-- ON A.Band_ID = B.Band_ID
-- JOIN Concerts C
-- ON A.Concert_ID = C.Concert_ID;

-- CREATE TABLE baseball_test (
-- game_id INT (11),
-- batter INT (11),
-- hit INT (11),
-- atBat INT (11),
-- PRIMARY KEY (game_id)
-- );

-- select 
--   year, 
--   sum(total_sales) as sum_of_year, 
--   avg(sum(total_sales)) over () as avg_sum
-- from sales_report 
-- group by year
-- order by sum(total_sales) desc
-- fetch first 1 row only;
-- SELECT EXTRACT(YEAR FROM CURRENT_DATE)

SELECT batter, SUM(Hit) AS total_h,SUM(atBat) as total_ab,(SUM(Hit) / SUM(atBat)) AS bAVG
FROM batter_counts  
GROUP BY batter
ORDER BY batter ASC;



SELECT bc.batter,bc.Hit, bc.atBat,g.game_id, g.local_date,(SUM(Hit) / SUM(atBat)) AS bAVG
FROM batter_counts bc 
JOIN game g
ON g.game_id = bc.game_id
GROUP BY EXTRACT(YEAR FROM g.local_date)
ORDER BY bc.batter desc;






