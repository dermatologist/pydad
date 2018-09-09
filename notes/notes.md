# Serializing models

* use mleap which is currently in version 0.8.1
* This does not support spark/pyspark 2.3
* I am using spark 2.2.0 and pyspark 2.2.0.post0
* The following environment variable needs to be set as below

```
export PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"

```
* Otherwise it gives a java port error
* I have added this to the spark batch file

* Following configuration is required

```python
   spark = SparkSession.builder. \
        appName("BellSparkTest1"). \
        config('spark.jars.packages',
               'ml.combust.mleap:mleap-spark-base_2.11:0.9.3,ml.combust.mleap:mleap-spark_2.11:0.9.3'). \
        config(conf=conf). \
        getOrCreate()

``` 
## Format of submission json
```
{
  "schema": {
    "fields": [{
      "name": "diagnosis",
      "type": "string"
    }, {
      "name": "bp",
      "type": "double"
    }, {
      "name": "other",
      "type": "string"
    }]
  },
  "rows": [["Fever", 2.0, "strict"]]
}

```

docker run -d --name mleap -p 8080:65327 dpsdce/mleap-serving_optimized:latest