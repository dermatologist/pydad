import logging

import findspark
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils

from src.pydad import __version__
from src.pydad.conf import ConfigParams


def main():
    _logger = logging.getLogger(__name__)
    findspark.init(ConfigParams.__SPARK_HOME__)
    SparkContext.setSystemProperty('spark.executor.memory', '32g')
    SparkContext.setSystemProperty('spark.driver.memory', '32g')

    sc = SparkContext(appName='SparkTest', master=ConfigParams.__MASTER_UI__)

    training_data = MLUtils.loadLibSVMFile(sc, "/home/beapen/scratch/libsvm")
    # Build the model
    model = LogisticRegressionWithLBFGS.train(training_data)

    # Evaluating the model on training data
    labelsAndPreds = training_data.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training_data.count())
    print("Training Error = " + str(trainErr))

    # Save and load model
    model.save(sc, "/home/beapen/scratch/pythonLogisticRegressionWithLBFGSModel")

    _logger.info("Script ends here")
    print(__version__)


def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("*")) for c in cols])


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
