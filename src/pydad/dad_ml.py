import logging

import findspark
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from src.pydad import __version__
from src.pydad.conf import ConfigParams


def main():
    _logger = logging.getLogger(__name__)
    findspark.init(ConfigParams.__SPARK_HOME__)
    SparkContext.setSystemProperty('spark.executor.memory', '4g')
    SparkContext.setSystemProperty('spark.driver.memory', '4g')

    sc = SparkContext(appName='SparkTest', master=ConfigParams.__MASTER_UI__)
    sqlContext = SQLContext(sc)

    df = sqlContext.read.csv(
        ConfigParams.__DAD_PATH__, header=True, mode="DROPMALFORMED"
    )

    vector_udf = F.udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))

    df = df.withColumn("morbidities", F.array("D_I10_1", "D_I10_2", "D_I10_3"))
    df = df.withColumn("treatments", F.array("I_CCI_1", "I_CCI_2", "I_CCI_3"))

    mindexer = CountVectorizer(inputCol="morbidities", outputCol="morbidityIndex")
    tindexer = CountVectorizer(inputCol="treatments", outputCol="treatmentIndex")

    pipeline = Pipeline().setStages([mindexer, tindexer])
    transformedDf = pipeline.fit(df).transform(df) \
        .select("AGRP_F_D", "GENDER", "WGHT_GRP",
                vector_udf('morbidityIndex').alias('morbidityIndex'),
                vector_udf('treatmentIndex').alias('treatmentIndex'))
    # to_print = transformedDf.collect()
    # print(to_print)
    sqlContext.clearCache()

    transformedDf.show(3)

    _logger.info("Script ends here")
    print(__version__)


def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("*")) for c in cols])


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
