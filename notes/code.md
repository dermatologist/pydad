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