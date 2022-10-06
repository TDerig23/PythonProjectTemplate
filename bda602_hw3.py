from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession


def main():
    appName = "App"
    master = "local[*]"
    spark = (
        SparkSession.builder.appName(appName)
        .master(master)
        .config(
            "spark.jars",
            "/mnt/c/Users/thoma/scr/PythonProjectTemplate/mariadb-java-client-3.0.8.jar",
        )
        .enableHiveSupport()
        .getOrCreate()
    )

    sql = (
        "SELECT nb.batter,nb.Hit,nb.atBat,nb.game_id,nb.local_date,"
        "SUM(nb.Hit) AS total_h,SUM(nb.atBat) as total_ab,(SUM(nb.Hit) / SUM(nb.atBat)) AS rolling_avg"
        "FROM new_baseball nb"
        "JOIN new_baseball nb2"
        "on nb.batter"
        "where nb.local_date between nb.local_date -100 and nb2.local_date"
        "GROUP by nb.batter,nb.local_date"
    )
    database = "baseball"
    user = "tderig"
    password = ""
    server = "127.0.0.1"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql)
        .option("user", user)
        .option("password", password)
        .option("trustServerCertificate", True)
        .option("driver", jdbc_driver)
        .load()
    )
    df.show(5)
    df.printSchema()


if __name__ == "__main__":
    main()
