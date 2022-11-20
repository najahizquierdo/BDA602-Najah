CREATE OR REPLACE TABLE pitchers_rolling (
    SELECT
        t.game_id
        , t.team_id
        , t.Hit/t.atBat as batting_avg
        , t.Grounded_Into_DP as double_play
        , t.stolenBase2B + t.stolenBase3B + t.stolenBaseHome as total_stolen
        , t.Strikeout as strikeout_total
        ,t.toBase as total_to_base
    FROM team_pitching_counts t
    JOIN inning i ON i.game_id = t.game_id
    LIMIT 0,3000
);

CREATE OR REPLACE TABLE away_stats(
    SELECT
        g.game_id
        , g.away_team_id as away_id
        , t.batting_avg as away_batting_avg
        , t.total_stolen as away_total_stolen
        , t.double_play AS away_double_play_opp
        , t.strikeout_total as away_strikeout_total
        , t.total_to_base as away_total_to_base
    FROM game g
    JOIN pitchers_rolling t ON g.away_team_id = t.team_id
    GROUP BY g.game_id
    );

CREATE OR REPLACE TABLE home_stats(
    SELECT
        g.game_id
        , g.home_team_id as home_id
        , t.batting_avg as home_batting_avg
        , t.total_stolen as home_total_stolen
        , t.double_play AS home_double_play_opp
        , t.strikeout_total as home_strikeout_total
        , t.total_to_base as home_total_to_base
    FROM game g
    JOIN pitchers_rolling t ON g.home_team_id = t.team_id
    GROUP BY g.game_id
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

CREATE OR REPLACE TABLE final_results(
    SELECT
        g.game_id as game_id
        , g.binary_winner as binary_winner
        , g.shutout_team as shutout_team
        , h.home_total_stolen as home_total_stolen
        , h.home_batting_avg as home_batting_avg
        , h.home_double_play_opp as home_slugging_avg
        , h.home_strikeout_total AS home_double_play_opp
        , h.home_total_to_base as home_strikeout_total
        , a.away_batting_avg as away_batting_avg
        , a.away_total_stolen as away_total_stolen
        , a.away_double_play_opp AS away_double_play_opp
        , a.away_strikeout_total as away_strikeout_total
        , a.away_total_to_base as away_total_to_base
        from game_stats g
        JOIN home_stats h ON h.game_id = g.game_id
        JOIN away_stats a on a.game_id = g.game_id
        GROUP BY game_id
);





#These were my original tables but the calculations never allowed my queries to work :\ but figured i would include them so you can see calculations i attempted
# CREATE OR REPLACE TABLE game_stats (
#     SELECT
#         t.game_id as game_id
#         , CASE
#             WHEN b.winner_home_or_away = 'H' THEN 0
#             ELSE 1 END AS binary_winner
#         , CASE
#             WHEN away_runs = 0 THEN 0
#             WHEN home_runs = 0 THEN 1
#             ELSE NULL END AS shutout_team
#         , b.winddir as wind_direction
#         FROM game g
#         JOIN boxscore b ON g.game_id = b.game_id
#         JOIN team_pitching_counts t ON g.game_id = t.game_id
#         GROUP BY t.game_id
# );
# CREATE OR REPLACE TABLE home_stats(
#     SELECT
#         g.game_id
#         , g.home_team_id
#         , (t.home_runs * 9) / i.inning as home_earned_run_avg
#         , (t.Strikeout + t.Walk) / i.inning as home_power_finesse
#         , t.Grounded_Into_DP AS home_double_play_opp
#         , t.Strikeout as home_strikeout_total
#         , (t.Strikeout/t.toBase) as home_strikeout_to_walk_ratio
#         , t.Hit/t.atBat as home_batting_avg
#         , (t.Hit+t.Walk+t.Hit_By_Pitch)/(t.atBat+t.Walk+t.Hit_By_Pitch+t.Sac_Fly) as home_on_base_percentage
#     FROM game g
#     JOIN inning i ON g.game_id = i.game_id
#     JOIN team_pitching_counts t ON g.home_team_id = t.team_id
#     group by g.game_id
#     );
#
#
# CREATE OR REPLACE TABLE away_stats(
#     SELECT
#         g.game_id
#         , g.away_team_id as away_id
#         , (t.homerun * 9) / i.inning as away_earned_run_avg
#         , (t.Strikeout + t.Walk) / i.inning as away_power_finesse
#         , t.double_play AS away_double_play_opp
#         , t.Strikeout as away_strikeout_total
#         , (t.Strikeout/t.total_to_base) as away_strikeout_to_walk_ratio
#         , t.Hit/t.total_to_base as away_batting_avg
#         , t.total_to_base as away_total_to_base
#     FROM game g
#     JOIN inning i ON g.game_id = i.game_id
#     JOIN pitchers_rolling t ON g.away_team_id = t.team_id
#     GROUP BY g.game_id
#     );