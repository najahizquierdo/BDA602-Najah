CREATE TABLE historical_average(
  select batter,
  cast(round(SUM(Hit)/SUM(atBat), 3) as float) AS batting_average
  from batter_counts
  group by batter
  order by batting_average desc
);

CREATE TABLE annual_average(
  select batter_counts.batter
  , EXTRACT(year from game.local_date) as batting_year
  , cast(round(NULLIF(SUM(batter_counts.Hit),0)/NULLIF(SUM(batter_counts.atBat),0), 3) as float) AS batting_average
  from batter_counts
  join game
  on batter_counts.game_id = game.game_id
  group by batter_counts.batter, batting_year asc
);
