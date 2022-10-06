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

CREATE TEMPORARY TABLE last100Games(
  select game_id
  , local_date
  from game
  order by local_date DESC LIMIT 100
);

CREATE OR REPLACE TEMPORARY TABLE rollingAverage(
  select batter_counts.batter
  , case when SUM(batter_counts.atBat) = 0 then 0
      else cast(SUM(batter_counts.Hit)/SUM(batter_counts.atBat) as float) end AS batting_average
  , batter_counts.game_id
  from batter_counts
  join last100Games
  on batter_counts.game_id = last100Games.game_id
  group by batter_counts.batter
  order by batter,last100Games.local_date, batting_average
);
