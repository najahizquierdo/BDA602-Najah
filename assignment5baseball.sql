USE BASEBALL;

CREATE OR REPLACE TABLE pitchers_rolling (
    SELECT
        game_id
        , team_id
        , Strikeout as Strikeout
        , Walk as Walk
        , Grounded_Into_DP as double_play
        , stolenBase2B + stolenBase3B + stolenBaseHome as total_stolen
        , toBase as total_to_base
    FROM team_pitching_counts
);

CREATE OR REPLACE TABLE game_stats(
    SELECT
        g.game_id as game_id
        , CASE
            WHEN b.winner_home_or_away = 'H' THEN 1
            WHEN b.winner_home_or_away = 'A' THEN 0
            ELSE NULL END AS binary_winner
        , CASE
            WHEN b.away_runs = 0 THEN 0
            WHEN b.home_runs = 0 THEN 1
            ELSE NULL END AS shutout_team
        FROM game g
        JOIN boxscore b ON g.game_id = b.game_id
        group by game_id
);
CREATE OR REPLACE TABLE away_stats(
    SELECT

        g.game_id
        , g.away_team_id
        , r.a_batting_avg
        , r.a_slugging_avg
        , t.total_stolen
        , t.total_to_base
        , t.Strikeout
        , t.Walk
        , t.double_play
        , r.a_strike_to_walk
        , r.a_ip
    FROM game g
    JOIN result r ON r.gameid = g.game_id
    JOIN pitchers_rolling t ON g.away_team_id = t.team_id AND t.game_id = g.game_id
    group by game_id
    );
CREATE OR REPLACE TABLE home_stats(
    SELECT
        g.game_id
        , g.home_team_id
        , r.h_batting_avg
        , r.h_slugging_avg
        , t.total_stolen
        , t.total_to_base
        , t.Strikeout
        , t.Walk
        , t.double_play
        , r.h_strike_to_walk
        , r.h_ip
    FROM game g
    JOIN result r ON r.gameid = g.game_id
    JOIN pitchers_rolling t ON g.home_team_id = t.team_id AND t.game_id = g.game_id
    );


CREATE OR REPLACE TABLE results(
    SELECT
        g.game_id as game_id
        , g.binary_winner as binary_winner
        , h.h_batting_avg as home_batting_avg
        , h.h_slugging_avg as home_slugging_avg
        , h.double_play AS home_double_play_opp
        , h.Strikeout as home_strikeout_total
        , h.total_stolen as home_bases_stolen
        , h.total_to_base as home_pitches_to_base
        , h.Walk as home_total_pitcher_walks
        , h.h_strike_to_walk as home_strikeout_to_walk_ratio
        , h.h_ip as home_innings_pitched
        , a.a_batting_avg as away_batting_avg
        , a.a_slugging_avg as away_slugging_avg
        , a.double_play AS away_double_play_opp
        , a.Strikeout as away_strikeout_total
        , a.total_stolen as away_bases_stolen
        , a.total_to_base as away_pitches_to_base
        , a.Walk as away_total_pitcher_walks
        , a.a_strike_to_walk as away_strikeout_to_walk_ratio
        , a.a_ip as away_innings_pitched
        from game_stats g
        JOIN home_stats h ON h.game_id = g.game_id
        JOIN away_stats a on a.game_id = g.game_id
);