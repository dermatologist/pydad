# Code

```python
    df = df.withColumn("morbidities", F.array("D_I10_1", "D_I10_2", "D_I10_3", "D_I10_4", "D_I10_5"))
    df = df.withColumn("treatments", F.array("I_CCI_1", "I_CCI_2", "I_CCI_3", "I_CCI_4", "I_CCI_5"))
    df.show(5)

    encoder = OneHotEncoderEstimator(inputCols=["morbidityIndex", "treatmentIndex"],
                                     outputCols=["categoryVec1", "categoryVec2"])
    model = encoder.fit(transformedDf)
    encoded = model.transform(transformedDf)
    encoded.show(5)

    indexer = CountVectorizer(inputCol="morbidities", outputCol="morbidityIndex")
    df = indexer.fit(df).transform(df)
    indexer = CountVectorizer(inputCol="treatments", outputCol="treatmentIndex")
    df = indexer.fit(df).transform(df)

    encoder = OneHotEncoder(inputCol="morbidityIndex", outputCol="morbidityVec")

    df = encoder.transform(df)

    df.createOrReplaceTempView("dad")
    sqlDF = sqlContext.sql("SELECT AGRP_F_D,GENDER,WGHT_GRP,morbidityIndex,treatmentIndex FROM dad")
    sqlDF.show(5)

    vector_udf = F.udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))

    sqlDF = sqlDF.select(vector_udf('morbidityIndex').alias('morbidityIndex'))

    sqlDF.show(5)

    df = sqlDF.rdd.map(lambda p:
                   Row(
                       age=p[0],
                       gender=p[1],
                       weight=p[2],
                       morbidity=p[3],
                       treatment=p[4],
                   )).toDF()
    df.show(5)
    df = df.withColumn("morbidities", myConcat("D_I10_1", "D_I10_2", "D_I10_3", "D_I10_4", "D_I10_5"))
    df = df.withColumn("treatments", myConcat("I_CCI_1", "I_CCI_2", "I_CCI_3", "I_CCI_4", "I_CCI_5"))

    df.show(5)

    indexer = StringIndexer(inputCol="treatments", outputCol="treatmentIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.show(5)

    df.createOrReplaceTempView("dad")
    sqlDF = sqlContext.sql("SELECT DISTINCT D_I10_1 FROM dad")
    sqlDF.show()

    indexer = StringIndexer(inputCol="D_I10_1", outputCol="categoryIndex")
    indexed = indexer.fit(sqlDF).transform(sqlDF)
    indexed.show()

    pairs = df.map(lambda x:
                   (x.split(" ")[0], x)

                   )


    indexer = StringIndexer(inputCol="D_I10_1", outputCol="categoryIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.show()

    cols = df.columns
    vectorAssembler = VectorAssembler(inputCols=cols, outputCol="features")
    vdf = vectorAssembler.transform(df)

    mat = RowMatrix(vdf)
    pc = mat.computePrincipalComponents(4)
    print("Principal Components \n")
    print(pc)

    projected = mat.multiply(pc)
    print("Projected \n")
    print(projected)
```

```python

   # like_f = F.udf(lambda col: True if 'ICDF' in col else False, BooleanType())
    # df.filter(like_f()).select().show(5)

    # df.show(5)
    # df.select("ICDF176").show(5)

    # vector_udf = F.udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
    #
    # df = df.withColumn("morbidities", F.array("D_I10_1", "D_I10_2", "D_I10_3"))
    # df = df.withColumn("treatments", F.array("I_CCI_1", "I_CCI_2", "I_CCI_3"))
    #
    # mindexer = CountVectorizer(inputCol="morbidities", outputCol="morbidityIndex")
    # tindexer = CountVectorizer(inputCol="treatments", outputCol="treatmentIndex")
    #
    # pipeline = Pipeline().setStages([mindexer, tindexer])
    # transformedDf = pipeline.fit(df).transform(df) \
    #     .select("AGRP_F_D", "GENDER", "WGHT_GRP",
    #             vector_udf('morbidityIndex').alias('morbidityIndex'),
    #             vector_udf('treatmentIndex').alias('treatmentIndex'))
    # # to_print = transformedDf.collect()
    # # print(to_print)
    # sqlContext.clearCache()
    #
    # transformedDf.show(3)

```


```python
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
```

## Redundant imports

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer

from pyspark.sql.types import *

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

from pyspark.ml.evaluation import MulticlassClassificationEvaluator



    # morbidity_columns = morbidity.schema.names
    # tlos_column = 'TLOS_CAT'
    #
    # print(morbidity_columns)
    # tlos.show(5)



```

https://stackoverflow.com/questions/43988801/pyspark-modify-column-values-when-another-column-value-satisfies-a-condition

```
from pyspark.sql.functions import *

df\
.withColumn('Id_New',when(df.Rank <= 5,df.Id).otherwise('other'))\
.drop(df.Id)\
.select(col('Id_New').alias('Id'),col('Rank'))\
.show()

```

## To explore

* https://spark.apache.org/docs/latest/ml-classification-regression.html
