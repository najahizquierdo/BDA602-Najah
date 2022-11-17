#this isnt implemented into the python code yet so you dont have to run this right now (only for file that i turn into the professor)
CREATE OR REPLACE TABLE game_stats (
    SELECT
        t.game_id as game_id
        , CASE
            WHEN b.winner_home_or_away = 'H' THEN 0
            ELSE 1 END AS binary_winner
        , CASE
            WHEN away_runs = 0 THEN 0
            WHEN home_runs = 0 THEN 1
            ELSE NULL END AS shutout_team
        , b.winddir as wind_direction
        FROM game g
        JOIN boxscore b ON g.game_id = b.game_id
        JOIN team_pitching_counts t ON g.game_id = t.game_id
        GROUP BY t.game_id
)
CREATE OR REPLACE TABLE home_stats(
    SELECT
        g.game_id
        , g.home_team_id
        , (b.home_runs * 9) / i.inning as home_earned_run_avg
        , (t.Strikeout + t.Walk) / i.inning as home_power_finesse
        , t.Grounded_Into_DP AS home_double_play_opp
        , t.Strikeout as home_strikeout_total
        , (bc.Strikeout/bc.toBase) as home_strikeout_to_walk_ratio
        , bc.Hit/bc.atBat as home_batting_avg
        , (bc.Hit+bc.Walk+bc.Hit_By_Pitch)/(bc.atBat+bc.Walk+bc.Hit_By_Pitch+bc.Sac_Fly) as home_on_base_percentage
    FROM game g
    JOIN inning i ON g.game_id = i.game_id
    JOIN team_batting_counts bc ON g.home_team_id = bc.team_id
    JOIN boxscore b ON g.home_team_id = bc.team_id
    JOIN team_pitching_counts t ON g.home_team_id = bc.team_id
    );
CREATE OR REPLACE TABLE away_stats(
    SELECT
        g.game_id
        , g.away_team_id
        , (b.away_runs * 9) / i.inning as away_earned_run_avg
        , (t.Strikeout + t.Walk) / i.inning as away_power_finesse
        , t.Grounded_Into_DP AS away_double_play_opp
        , t.Strikeout as away_strikeout_total
        , (bc.Strikeout/bc.toBase) as away_strikeout_to_walk_ratio
        , bc.Hit/bc.atBat as away_batting_avg
        , (bc.Hit+bc.Walk+bc.Hit_By_Pitch)/(bc.atBat+bc.Walk+bc.Hit_By_Pitch+bc.Sac_Fly) as away_on_base_percentage
    FROM game g
    JOIN inning i ON g.game_id = i.game_id
    JOIN team_batting_counts bc ON g.away_team_id = bc.team_id
    JOIN boxscore b ON g.away_team_id = bc.team_id
    JOIN team_pitching_counts t ON g.away_team_id = bc.team_id
    LIMIT 1
    );