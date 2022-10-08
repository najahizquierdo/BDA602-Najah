import sys
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.ml.feature import SQLTransformer

def main():
    appName = "Baseball Scripts in PySpark"
    spark = SparkSession.builder.master("local[*]").appName(appName).getOrCreate()

    url = "jdbc:mysql://localhost:3306/baseball?user=spark&password=kontext"

    driver = "org.mariadb.jdbc.Driver"
    user = "spark"
    password = "kontext"

    batters_game = spark.sql(
        """
          select
          b.batter
          , b.game_id
          , b.atBat
          , b.Hit
          , g.local_date
          from batter_counts b
          join game g
          on b.game_id = g.game_id
        """
    )
    batters_game.createOrReplaceTempView("batters_game")
    batters_game.persist(StorageLevel.MEMORY_AND_DISK)

    t_last100 = spark.sql(
    """
      select b.game_id
      , b.local_date
      from batters_game b
      where b.local_date between DATE_ADD(b.local_date, interval -100 day) and DATE_ADD(b.local_date, interval 0 day)
      group by b.local_date
      order by b.game_id
    """
    )

    t_last100.createOrReplaceTempView("t_last100")
    t_last100.persist(StorageLevel.MEMORY_AND_DISK)

    last100_rolling_avg = spark.sql(
    """
      select b.batter
      , b.local_date
      , case when SUM(b.atBat) = 0 then 0
          else cast(SUM(b.Hit)/SUM(b.atBat) as float) end AS batting_average
      from batters_game b
      join last100Games l
      on l.game_id = b.game_id
      group by b.batter
      order by b.batter, batting_average
    """
    )

    last100_rolling_avg.createOrReplaceTempView("last100_rolling_avg")
    last100_rolling_avg.persist(StorageLevel.MEMORY_AND_DISK)

    baseballgame = (
        spark.read.format("jdbc")
        .option("url", url)
        .option("dbtable", "baseball.game")
        .option("user", user)
        .option("password", password)
        .option("driver", driver)
        .load()
    )
    baseballgame.show()
    baseballgame.createOrReplaceTempView("baseballgame")
    baseballgame.persist(StorageLevel.MEMORY_AND_DISK)

    batter_counts = (
        spark.read.format("jdbc")
        .option("url", url)
        .option("dbtable", "baseball.batter_counts")
        .option("user", user)
        .option("password", password)
        .option("driver", driver)
        .load()
        )
    batter_counts.show()
    batter_counts.createOrReplaceTempView("batter_counts")
    batter_counts.persist(StorageLevel.MEMORY_AND_DISK)

    sqlTrans = SQLTransformer().setStatement(last100_rolling_avg)
    sqlTrans.transform(batters_game).show()

if __name__ == "__main__":
    sys.exit(main())
