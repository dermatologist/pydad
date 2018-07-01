import logging

import findspark
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SQLContext

from src.pydad import __version__
from src.pydad.conf import ConfigParams


def main():
    _logger = logging.getLogger(__name__)
    findspark.init(ConfigParams.__SPARK_HOME__)
    SparkContext.setSystemProperty('spark.executor.memory', '3g')
    SparkContext.setSystemProperty('spark.driver.memory', '3g')

    sc = SparkContext(appName='SparkTest', master=ConfigParams.__MASTER_UI__)
    sqlContext = SQLContext(sc)

    df = sqlContext.read.csv(
        ConfigParams.__DAD_PATH__, header=True, mode="DROPMALFORMED"
    )

    df = df.withColumn("morbidities", F.array("D_I10_1", "D_I10_2", "D_I10_3", "D_I10_4", "D_I10_5"))
    df = df.withColumn("treatments", F.array("I_CCI_1", "I_CCI_2", "I_CCI_3", "I_CCI_4", "I_CCI_5"))
    # df.show(5)

    indexer = CountVectorizer(inputCol="morbidities", outputCol="morbidityIndex")
    df = indexer.fit(df).transform(df)
    indexer = CountVectorizer(inputCol="treatments", outputCol="treatmentIndex")
    df = indexer.fit(df).transform(df)

    # encoder = OneHotEncoder(inputCol="morbidityIndex", outputCol="morbidityVec")
    #
    # df = encoder.transform(df)

    df.show(5)

    # df = df.withColumn("morbidities", myConcat("D_I10_1", "D_I10_2", "D_I10_3", "D_I10_4", "D_I10_5"))
    # df = df.withColumn("treatments", myConcat("I_CCI_1", "I_CCI_2", "I_CCI_3", "I_CCI_4", "I_CCI_5"))
    #
    # df.show(5)
    #
    # indexer = StringIndexer(inputCol="treatments", outputCol="treatmentIndex")
    # indexed = indexer.fit(df).transform(df)
    # indexed.show(5)

    # df.createOrReplaceTempView("dad")
    # sqlDF = sqlContext.sql("SELECT DISTINCT D_I10_1 FROM dad")
    # sqlDF.show()
    #
    # indexer = StringIndexer(inputCol="D_I10_1", outputCol="categoryIndex")
    # indexed = indexer.fit(sqlDF).transform(sqlDF)
    # indexed.show()

    # pairs = df.map(lambda x:
    #                (x.split(" ")[0], x)
    #
    #                )

    #
    # indexer = StringIndexer(inputCol="D_I10_1", outputCol="categoryIndex")
    # indexed = indexer.fit(df).transform(df)
    # indexed.show()

    # cols = df.columns
    # vectorAssembler = VectorAssembler(inputCols=cols, outputCol="features")
    # vdf = vectorAssembler.transform(df)

    # mat = RowMatrix(vdf)
    # pc = mat.computePrincipalComponents(4)
    # print("Principal Components \n")
    # print(pc)
    #
    # projected = mat.multiply(pc)
    # print("Projected \n")
    # print(projected)

    _logger.info("Script ends here")
    print(__version__)


def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("*")) for c in cols])


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
