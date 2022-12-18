import sys
import pandas as pd
import sqlalchemy
import pymysql
from sklearn.model_selection import train_test_split
from predfeatures import set_html_up, correlation_metrics, brute_force, mean_of_response, cont_plots, logreg, feature_importance, knn
#split @ date
def main():

    db_user = "root"
    db_pass = "password"
    db_host = "db"
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
    df = pd.read_sql_query(results, sql_engine)
    df = df.dropna()
    response = 'game_winner'
    bin_response = 'binary_winner'
    numerical_cols = df._get_numeric_data().columns.drop("game_id")
    numerical_cols = numerical_cols.drop(bin_response)
    X = df[numerical_cols]
    y = df['game_winner'].dropna()
    y_bin = df['binary_winner'].dropna()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    set_html_up()
    correlation_metrics(df)
    logreg(X_test, y_test, X, y_bin, numerical_cols)
    cont_plots(df, numerical_cols, response)
    brute_force(X,y_bin)
    mean_of_response(X, y_bin)
    feature_importance(X, y, numerical_cols)
    knn(X_test, X_train, y_train, y_test, X, y)


    print(df.head())


if __name__ == "__main__":
    sys.exit(main())
