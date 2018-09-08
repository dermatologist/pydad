import logging

import findspark
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.sql import SQLContext
# This is not recognized by IntelliJ!, but still works.
from pyspark.sql.functions import col

from src.pydad import __version__
from src.pydad.conf import ConfigParams


def main():
    _logger = logging.getLogger(__name__)
    findspark.init(ConfigParams.__SPARK_HOME__)
    SparkContext.setSystemProperty('spark.executor.memory', '48g')
    SparkContext.setSystemProperty('spark.driver.memory', '6g')

    sc = SparkContext(appName='SparkTest', master=ConfigParams.__MASTER_UI__)
    sqlContext = SQLContext(sc)

    df = sqlContext.read.csv(
        ConfigParams.__DAD_PATH__, header=True, mode="DROPMALFORMED"
    )

    # String type converted to float type.
    df = df.select(*(col(c).cast("float").alias(c) for c in df.columns))

    # Change all NA to 0
    df = df.na.fill(0)

    # Recode TLOS_CAT to binary
    df = df \
        .withColumn('TLOS_CAT_NEW', F.when(df.TLOS_CAT <= 5, 0).otherwise(1)) \
        .drop(df.TLOS_CAT)

    df.show(3)

    # Prints number of columns. 596 is the new binary variable TLOS_CAT_NEW
    print(len(df.columns))  # 596

    transformed_df = df.select(df.columns[155:]).rdd.map(lambda row: LabeledPoint(row[440], Vectors.dense(row[6:-1])))

    print(transformed_df.take(5))

    MLUtils.saveAsLibSVMFile(transformed_df, "/home/beapen/scratch/libsvm")
    # transformed_df.write.format("libsvm").save("/home/beapen/scratch/libsvm")

    _logger.info("Script ends here")
    print(__version__)


def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("*")) for c in cols])


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
