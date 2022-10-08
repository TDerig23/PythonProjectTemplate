from pyspark.sql import SparkSession
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import Pipeline, Transformer


class SplitColumnTransform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(SplitColumnTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        return dataset.show()


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
        """SELECT bc.batter,bc.Hit, bc.atBat,g.game_id, g.local_dateFROM batter_counts bc,
        SUM(nb.Hit) AS total_h,SUM(nb.atBat) as total_ab,(SUM(nb.Hit) / SUM(nb.atBat)) AS rolling_avg
    JOIN game g
    ON g.game_id = bc.game_id
    order by bc.batter, bc.game_id"""

    )
    database = "baseball"
    user = ""
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

    df.createOrReplaceTempView("rolling_avg")
    df2 = spark.sql("""select batter, game_id, SUM(Hit) AS total_h,SUM(nb.atBat) 
                    as total_ab,(SUM(nb.Hit) / SUM(nb.atBat)) AS rolling_avg 
                    where nb.local_date >= 2012-03-20 00:00:00.000  and nb2.local_date < 2012-06-28 22:15:00.000 
                    GROUP by nb.batter,nb.local_date"""
                      )



    new_transform = SplitColumnTransform()
    pipeline = Pipeline(stages=[new_transform])
    model = pipeline.fit(df2)
    model.transform(df2)


if __name__ == "__main__":
    main()
#
