import sys

import pandas
import sqlalchemy

from midterm import df_processing


def main():
    db_user = "root"
    db_pass = "x11docker"
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    results = """
            SELECT * 
            FROM final_results 
            group by game_id
    """
    game_df = pandas.read_sql_query(results, sql_engine)
    print(game_df.head())
    df_processing(game_df, "binary_winner")


if __name__ == "__main__":
    sys.exit(main())
