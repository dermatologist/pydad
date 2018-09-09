import logging
import uuid

import findspark
import pyspark.sql.functions as F
from pyspark.conf import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
# This is not recognized by IntelliJ!, but still works.
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer

from src.pydad import __version__
from src.pydad.conf import ConfigParams

# Imports MLeap serialization functionality for PySpark

"""
Ref: https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
"""


def main():
    _logger = logging.getLogger(__name__)
    findspark.init(ConfigParams.__SPARK_HOME__)

    # Configuration
    conf = SparkConf(). \
        setAppName('BellSpark')
    # Spark Session replaces SparkContext
    spark = SparkSession.builder. \
        appName("BellSparkTest1"). \
        config('spark.jars.packages',
               'ml.combust.mleap:mleap-spark-base_2.11:0.9.3,ml.combust.mleap:mleap-spark_2.11:0.9.3'). \
        config(conf=conf). \
        getOrCreate()

    # Read csv
    df = spark.read.csv(ConfigParams.__DAD_PATH__, header=True, inferSchema=True)

    # Select the columns that we need
    df = df.select('TLOS_CAT', 'ACT_LCAT', 'ALC_LCAT', \
                    'ICDCOUNT', 'CCICOUNT')

    # Change all NA to 0
    df = df.na.fill(0)

    # Recode TLOS_CAT to binary
    df = df \
        .withColumn('TLOS_CAT_NEW', F.when(df.TLOS_CAT <= 5, 0).otherwise(1)) \
        .drop(df.TLOS_CAT)

    df.printSchema()

    stages = []

    input_column_names = ['ACT_LCAT', 'ALC_LCAT', 'ICDCOUNT', 'CCICOUNT']

    feature_assembler = VectorAssembler(inputCols=input_column_names, outputCol="features")

    stages += [feature_assembler]

    # # Train and Test
    # train, test = df.randomSplit([0.7, 0.3], seed=2018)
    # print("Training Dataset Count: " + str(train.count()))
    # print("Test Dataset Count: " + str(test.count()))

    lr = LogisticRegression(featuresCol='features', labelCol='TLOS_CAT_NEW', maxIter=10)

    stages += [lr]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    predictions = pipeline_model.transform(df)


    # Predict
    predictions.select('TLOS_CAT_NEW', 'prediction').show(100)

    # Filename
    random_file = "jar:file:/home/beapen/scratch/pyspark" + str(uuid.uuid4()) + ".zip"

    # Serialize
    pipeline_model.serializeToBundle(random_file, pipeline_model.transform(df))

    _logger.info("Script ends here")
    print(__version__)


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
