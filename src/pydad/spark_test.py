import logging

import findspark
from pyspark import SparkContext
from pyspark.sql import SQLContext

from src.pydad import __version__
from src.pydad.conf import ConfigParams


def main():
    _logger = logging.getLogger(__name__)
    findspark.init(ConfigParams.__SPARK_HOME__)
    sc = SparkContext(appName='SparkTest', master=ConfigParams.__MASTER_UI__)
    sqlContext = SQLContext(sc)

    df = sqlContext.read.csv(
        ConfigParams.__DAD_PATH__, header=True, mode="DROPMALFORMED"
    )

    print(df.take(5))
    _logger.info("Script ends here")
    print(__version__)


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
