CREATE TABLE historicalAverage(
  select batter
  , case when SUM(batter_counts.atBat) = 0 then 0
    else cast(SUM(batter_counts.Hit)/SUM(batter_counts.atBat) as float) end AS batting_average
  from batter_counts
  group by batter
  order by batting_average desc
);

CREATE TABLE annualAverage(
  select batter_counts.batter
  , EXTRACT(year from game.local_date) as batting_year
  , case when SUM(batter_counts.atBat) = 0 then 0
    else cast(SUM(batter_counts.Hit)/SUM(batter_counts.atBat) as float) end AS batting_average
  from batter_counts
  join game
  on batter_counts.game_id = game.game_id
  group by batter_counts.batter, batting_year asc
);


CREATE OR REPLACE TEMPORARY TABLE batters_game (
  select
  b.batter
  , b.game_id
  , b.atBat
  , b.Hit
  , g.local_date
  from batter_counts b
  join game g
  on b.game_id = g.game_id
);

CREATE OR REPLACE TEMPORARY TABLE last100Games(
  select b.game_id
  , b.local_date
  from batters_game b
  where b.local_date between DATE_ADD(b.local_date, interval -100 day) and DATE_ADD(b.local_date, interval 0 day)
  group by b.local_date
  order by b.game_id
);


CREATE OR REPLACE TABLE last100_rolling_avg (
  select b.batter
  , b.local_date
  , case when SUM(b.atBat) = 0 then 0
      else cast(SUM(b.Hit)/SUM(b.atBat) as float) end AS batting_average
  from batters_game b
  join last100Games l
  on l.game_id = b.game_id
  group by b.batter
  order by b.batter, batting_average
);
