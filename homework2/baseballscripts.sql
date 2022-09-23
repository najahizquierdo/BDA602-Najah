CREATE TABLE historicalAverage(
  select batter
  , cast(round(NULLIF(SUM(batter_counts.Hit),0)/NULLIF(SUM(batter_counts.atBat),0), 3) as float) AS batting_average
  from batter_counts
  group by batter
  order by batting_average desc
);

CREATE TABLE annualAverage(
  select batter_counts.batter
  , EXTRACT(year from game.local_date) as batting_year
  , cast(round(NULLIF(SUM(batter_counts.Hit),0)/NULLIF(SUM(batter_counts.atBat),0), 3) as float) AS batting_average
  from batter_counts
  join game
  on batter_counts.game_id = game.game_id
  group by batter_counts.batter, batting_year asc
);

CREATE TEMPORARY TABLE last100Games(
  select game_id
  , local_date
  from game
  where local_date >= (DATE(NOW()) - INTERVAL 100 DAY)
  order by local_date DESC
);

CREATE TABLE rollingAverage(
  select batter_counts.batter
  , cast(round(NULLIF(SUM(batter_counts.Hit),0)/NULLIF(SUM(batter_counts.atBat),0), 3) as float) AS batting_average
  , batter_counts.game_id
  from batter_counts
  join last100Games
  on batter_counts.game_id = last100Games.game_id
  group by batter_counts.batter
  order by batter,last100Games.local_date, batting_average
);
