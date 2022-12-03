#!/usr/bin/env bash

sleep 10
echo "creating baseball..."
mysql -h najah-db -u root -ppassword -e "CREATE DATABASE IF NOT EXISTS baseball"
echo "loading sql database..."
mysql -h najah-db -u root -ppassword baseball < ./baseball.sql
echo "using baseball..."
mysql -h najah-db -u root -ppassword -e "USE baseball;"
echo "Find batting averages for all ... "

mysql -h najah-db -u root -ppassword -e '
USE baseball;
CREATE OR REPLACE TABLE batters_game
  select
  b.batter
  , b.game_id
  , case when b.atBat = 0 then 0
  else b.Hit/b.atBat end AS batting_average
  , g.local_date
  from batter_counts b
  join game g
  on b.game_id = g.game_id
  order by game_id;'

echo "Find batter info in a specific game..."
mysql -h najah-db -u root -ppassword -e '
USE baseball;
CREATE OR REPLACE TABLE specific_game
  select
  b.batter
  , g.local_date
  , b.batting_average
  , b.game_id
  from batters_game b
  join game g on b.game_id = g.game_id
  where b.game_id = 12560
  order by b.batter;'

echo "combining specific game and batting info..."
mysql -h najah-db -u root -ppassword -e '
USE baseball;
CREATE OR REPLACE TABLE batting_avg_game
  select DISTINCT
  s.batter
  , s.batting_average
  , s.game_id
  from specific_game s
  order by s.batting_average ASC;'

echo "Find rolling batting avg..."
mysql -h najah-db -u root -ppassword -e '
  USE baseball;
  CREATE OR REPLACE TABLE rolling_avg_100
    select DISTINCT
    b.batter
    , b.batting_average
    from batters_game b
    where b.local_date between "2011-01-01 00:00:00" AND "2012-01-01 00:00:00"
    order by b.batter;'

mysql -h najah-db -u root -ppassword -e 'USE baseball; select * from batting_avg_game;' > /app/results/specificgamebatavg.txt
mysql -h najah-db -u root -ppassword -e 'USE baseball; select * from rolling_avg_100;' > /app/results/battingavg.txt
