import sys

import pandas
import sqlalchemy

from midterm import df_processing


def main():
    db_user = "root"
    db_pass = "x11docker"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    pitching = """
                    SELECT
                t.game_id
                , CASE
                    WHEN b.winner_home_or_away = 'A' THEN winner_home_or_away = 0
                    ELSE b.winner_home_or_away = 1 END AS binary_winner
                , (b.away_runs*9)/i.inning as away_earned_run_avg
                , (b.home_runs*9)/i.inning as home_earned_run_avg
                , (t.Strikeout+t.Walk)/i.inning as power_finesse
                , ((t.stolenBase2B+t.stolenBase3B+t.stolenBaseHome)
                /(t.caughtStealing2B+t.caughtStealing3B+t.caughtStealingHome)) as stolen_base_percentage
                , t.Grounded_Into_DP AS double_play_opportunities
                , t.Strikeout as strikeout_total
                , CASE
                    WHEN away_runs = 0 THEN 0
                    WHEN home_runs = 0 THEN 1
                    ELSE NULL END AS shutout_team
                , (bc.Strikeout/bc.toBase) as strikeout_to_walk_ratio
                , b.winddir as wind_direction
                , bc.Hit/bc.atBat as batting_avg
                , bc.Groundout/bc.Flyout as ground_fly_ratio
                , (bc.Hit+bc.Walk+bc.Hit_By_Pitch)/(bc.atBat+bc.Walk+bc.Hit_By_Pitch+bc.Sac_Fly) as on_base_percentage
            FROM game g
            JOIN inning i ON g.game_id = i.game_id
            JOIN team_batting_counts bc ON g.game_id = bc.game_id
            JOIN boxscore b ON g.game_id = b.game_id
            JOIN team_pitching_counts t ON g.game_id = t.game_id
            GROUP BY t.game_id
    """

    # NEW QUERY I AM WORKING ON BECAUSE THE OTHER ONE IS WRONg :|
    # SELECT
    #     t.game_id
    #     ,CASE
    #         WHEN b.winner_home_or_away = 'H' THEN 0
    #         ELSE 1 END AS binary_winner
    #     , CASE
    #         WHEN away_runs = 0 THEN 0
    #         WHEN home_runs = 0 THEN 1
    #         ELSE NULL END AS shutout_team
    #     , b.winddir as wind_direction
    #     , h.home_earned_run_avg
    #     , h.home_power_finesse
    #     , h.home_double_play_opp
    #     , h.home_strikeout_total
    #     , h.home_strikeout_to_walk_ratio
    #     , h.home_batting_avg
    #     , h.home_on_base_percentage
    #     , a.away_team_id
    #     , a.away_earned_run_avg
    #     , a.away_power_finesse
    #     , a.away_double_play_opp
    #     , a.away_strikeout_total
    #     , a.away_strikeout_to_walk_ratio
    #     , a.away_batting_avg
    #     , a.away_on_base_percentage
    # FROM game_stats g
    # JOIN home_stats h ON g.game_id = h.game_id
    # JOIN away_stats a ON g.game_id = a.game_id
    # GROUP BY g.game_id
    baseball_df = pandas.read_sql_query(pitching, sql_engine)
    response = baseball_df["binary_winner"]
    df_processing(baseball_df, response)


# split 20/80 historically, pick a date

if __name__ == "__main__":
    sys.exit(main())
