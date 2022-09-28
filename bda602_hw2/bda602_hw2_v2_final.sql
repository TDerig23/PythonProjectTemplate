-- Baseball ⚾
-- Load this dataset into SQL
-- Calculate the batting average using SQL queries for every player
-- Annual, Historic 
-- Rolling (over last 100 days) 😓
-- Look at the last 100 days that player was in prior to this game
-- Store all these results in their own tables
-- Hints
-- Look at the tables already available: batter_counts, battersInGame
-- Do it for a single batter first
-- JOIN ing a table with itself may prove helpful
-- It's ok to make temporary tables for intermediary results

create index idx_batter_counts on baseball.batter_counts(game_id);
create index idx_game on baseball.game(game_id);

	-- Historical Average
-- SELECT batter, atBat, Hit,SUM(Hit) AS total_h,SUM(atBat) as total_ab,(SUM(Hit) / SUM(atBat)) AS bAVG
-- FROM batter_counts  
-- GROUP BY batter
-- ORDER BY batter ASC;

-- average by year 
-- SELECT bc.batter,bc.Hit, bc.atBat,g.game_id, g.local_date,YEAR(g.local_date) as year,
-- SUM(Hit) AS total_h,SUM(atBat) as total_ab,(SUM(Hit) / SUM(atBat)) AS yearly_bAVG
-- FROM batter_counts bc 
-- JOIN game g
-- ON g.game_id = bc.game_id
-- -- WHERE batter = 110029
-- GROUP BY year,batter
-- ORDER BY batter ASC;


-- rolling average


-- CREATE TABLE new_baseball
-- SELECT bc.batter,bc.Hit, bc.atBat,g.game_id, g.local_date 
-- FROM batter_counts bc 
-- JOIN game g
-- ON g.game_id = bc.game_id;

-- select MAX(local_date)
-- from game ;

SELECT nb.batter,nb.Hit,nb.atBat,nb.game_id,nb.local_date,
SUM(nb.Hit) AS total_h,SUM(nb.atBat) as total_ab,(SUM(nb.Hit) / SUM(nb.atBat)) AS rolling_avg
FROM new_baseball nb 
JOIN new_baseball nb2  
on nb.local_date 
where nb.local_date 
GROUP by nb.batter;




-- run when done 
-- DROP TABLE new_baseball;



-- ALTER TABLE new_baseball 
-- MODIFY Column Day Day;
-- UPDATE baseball_temp
-- set new_baseball.day = DAY(local_date);


-- SELECT batter,hit,atBat,game_id,local_date, 
-- (SUM(Hit) / SUM(atBat)) 
-- From new_baseball 
-- ORDER by local_date;


-- ALTER TABLE baseball_temp 
-- MODIFY Column Year year;
-- UPDATE baseball_temp
-- set baseball_temp.year = Year(local_date);


-- SELECT bt.batter,bt.Hit, bt.atBat,EXTRACT(YEAR FROM bt.local_date) as year,SUM(bt.Hit) AS 
-- total_h,SUM(bt.atBat) as total_ab,(SUM(Hit) / SUM(atBat)) AS bAVG
-- FROM baseball_temp bt
-- GROUP BY bt.batter;



