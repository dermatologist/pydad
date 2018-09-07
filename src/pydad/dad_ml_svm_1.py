import logging
from time import *

import findspark
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
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

    RANDOM_SEED = 13579
    TRAINING_DATA_RATIO = 0.7

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

    splits = [TRAINING_DATA_RATIO, 1.0 - TRAINING_DATA_RATIO]
    training_data, test_data = transformed_df.randomSplit(splits, RANDOM_SEED)

    print("Number of training set rows: %d" % training_data.count())
    print("Number of test set rows: %d" % test_data.count())

    start_time = time()

    model = SVMWithSGD.train(training_data, iterations=10)

    end_time = time()
    elapsed_time = end_time - start_time
    print("Time to train model: %.3f seconds" % elapsed_time)

    predictions = model.predict(test_data.map(lambda x: x.features))
    labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
    acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
    print("Model accuracy: %.3f%%" % (acc * 100))

    # Save model
    model.save(sc, "/home/beapen/scratch/pythonSVMModel")

    _logger.info("Script ends here")
    print(__version__)


def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("*")) for c in cols])


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
