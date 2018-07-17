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

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import RandomForest
from time import *
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col

# http://jarrettmeyer.com/2017/05/04/random-forests-with-pyspark
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

    # labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
    #
    # featureIndexer = \
    #     VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)
    #
    # # Split the data into training and test sets (30% held out for testing)
    # (trainingData, testData) = df.randomSplit([0.7, 0.3])
    #
    # # Train a RandomForest model.
    # rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    #
    # # Convert indexed labels back to original labels.
    # labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
    #                                labels=labelIndexer.labels)
    #
    # # Chain indexers and forest in a Pipeline
    # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    #
    # # Train model.  This also runs the indexers.
    # model = pipeline.fit(trainingData)
    #
    # # Make predictions.
    # predictions = model.transform(testData)
    #
    # # Select example rows to display.
    # predictions.select("predictedLabel", "label", "features").show(5)
    #
    # # Select (prediction, true label) and compute test error
    # evaluator = MulticlassClassificationEvaluator(
    #     labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    # accuracy = evaluator.evaluate(predictions)
    # print("Test Error = %g" % (1.0 - accuracy))
    #
    # rfModel = model.stages[2]
    # print(rfModel)  # summary only
    # # $example off$



    # df.select(df.columns[154:]).show(5)

    tlos = df.select(df.columns[154:155])
    morbidity = df.select(df.columns[161:])

    RANDOM_SEED = 13579
    TRAINING_DATA_RATIO = 0.7
    RF_NUM_TREES = 3
    RF_MAX_DEPTH = 4
    RF_MAX_BINS = 32

    df = df.select(*(col(c).cast("float").alias(c) for c in df.columns))

    df = df.na.fill(0)

    transformed_df = df.select(df.columns[154:]).rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[5:-1])))

    splits = [TRAINING_DATA_RATIO, 1.0 - TRAINING_DATA_RATIO]
    training_data, test_data = transformed_df.randomSplit(splits, RANDOM_SEED)

    print("Number of training set rows: %d" % training_data.count())
    print("Number of test set rows: %d" % test_data.count())

    start_time = time()

    model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={}, \
                                         numTrees=RF_NUM_TREES, featureSubsetStrategy="auto", impurity="gini", \
                                         maxDepth=RF_MAX_DEPTH, maxBins=RF_MAX_BINS, seed=RANDOM_SEED)

    end_time = time()
    elapsed_time = end_time - start_time
    print("Time to train model: %.3f seconds" % elapsed_time)

    predictions = model.predict(test_data.map(lambda x: x.features))
    labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
    acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
    print("Model accuracy: %.3f%%" % (acc * 100))

    start_time = time()

    metrics = BinaryClassificationMetrics(labels_and_predictions)
    print("Area under Precision/Recall (PR) curve: %.f" % (metrics.areaUnderPR * 100))
    print("Area under Receiver Operating Characteristic (ROC) curve: %.3f" % (metrics.areaUnderROC * 100))

    end_time = time()
    elapsed_time = end_time - start_time
    print("Time to evaluate model: %.3f seconds" % elapsed_time)

    _logger.info("Script ends here")
    print(__version__)


def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("*")) for c in cols])

if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
