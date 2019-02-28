# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run ../0-utils/0-Base.py

# +
import pyspark.sql.types as T
import pyspark.sql.functions as F

spark.conf.set("spark.sql.shuffle.partitions", 16)
spark.conf.set("spark.sql.execution.arrow.enabled", True)
# -

# Load the data about historical and eval period transactions:

df = spark.read.csv("../data/raw/*_transactions.csv", header=True)
df.printSchema()
display(df)

# +
dicts = sc.broadcast(dict([('Y', 1), ('N', 0)]))

from pyspark.sql import functions as F
from pyspark.sql import types as T

def newCols(x):
    return dicts.value[x]

callnewColsUdf = F.udf(newCols, T.IntegerType())

df = df.withColumn("authorized_flag_indexed", callnewColsUdf(F.col("authorized_flag")))

# +
# df = df.withColumn("installments", F.col("installments").cast(T.IntegerType()))

callNewColsUdf = F.udf(lambda x: 999 if x == '-1' else int(x), T.IntegerType())
df = df.withColumn("installments_fixed", callNewColsUdf(F.col("installments")))

# +
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql.functions import isnan

# dropLast=True set by default. That produce value 999 to vector of zeros [0]. 
# На самом деле я без понятия, но мне так кажется
encoder = OneHotEncoderEstimator(inputCols=["installments_fixed"],
                                 outputCols=["installments_vec"])
model = encoder.fit(df)
df = model.transform(df)
df.printSchema()

# +
# from pyspark.ml.stat import Summarizer

# # agg for vector column
# df = df.select("card_id", "month_lag", "installments_vec")
# feature_set = df.groupBy("card_id", "month_lag").agg(Summarizer.mean(df.installments_vec)).show()

# +
# %%time

# agg for transactions count
from pyspark.sql.functions import pandas_udf, PandasUDFType

df = df = df.select("card_id", "month_lag", "purchase_amount")
df = df.withColumn("month_lag", F.col("month_lag").cast(T.IntegerType()))
df = df.withColumn("purchase_amount", F.col("purchase_amount").cast(T.DoubleType()))

count_transactions_set = df.groupBy("card_id", "month_lag").agg(
    {'purchase_amount': "count"}).orderBy("card_id", "month_lag")
count_transactions_set.show()

# +
from pyspark.ml.stat import Summarizer

df = df.select("card_id", "month_lag", "purchase_amount", "installments_vec", "authorized_flag_indexed")
df = df.withColumn("month_lag", F.col("month_lag").cast(T.IntegerType()))
df = df.withColumn("authorized_flag_indexed", F.col("authorized_flag_indexed").cast(T.IntegerType()))
df = df.withColumn("purchase_amount", F.col("purchase_amount").cast(T.DoubleType()))
# -

# Calculate features and save them:

# +
# %%time

agg = (F.min("purchase_amount"),
       F.avg("purchase_amount"),
       F.max("purchase_amount"),
       F.avg("authorized_flag_indexed"))

feature_set = df.groupBy("card_id", "month_lag").agg(
    Summarizer.mean(df.installments_vec),
    *agg).orderBy("card_id", "month_lag")

# feature_set.write.format("csv") \
#            .mode("overwrite") \
#            .option("header", True) \
#            .save("../data/1-feature-engineered")

feature_set.show()
# -

dataframe = feature_set.toPandas()

display(dataframe)

dataframe.to_feather("../data/1-feature-engineered/feather.csv")

# +
from pyspark.ml.stat import Summarizer
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

dfs = sc.parallelize([Row(weight=1.0, features=Vectors.dense(1, 1, 1, 2, 2)),
                      Row(weight=1.0, features=Vectors.dense(2, 2, 2, 2, 2)),
                      Row(weight=0.0, features=Vectors.dense(4, 4, 4, 4, 5)),
                     Row(weight=0.0, features=Vectors.dense(1, 2, 3, 4, 5))]).toDF(["col1", "col2"])

dfs.show()
# create summarizer for multiple metrics "mean" and "count"
summarizer = Summarizer.metrics("mean", "count")


dfs.groupby("col2").agg(Summarizer.mean(dfs.col1)).show()

# # compute statistics for multiple metrics with weight
# dfs.select(summarizer.summary(df.features, df.weight)).show(truncate=False)

# # compute statistics for multiple metrics without weight
# dfs.select(summarizer.summary(df.features)).show(truncate=False)

# # compute statistics for single metric "mean" with weight
# dfs.select(Summarizer.mean(df.features, df.weight)).show(truncate=False)

# # compute statistics for single metric "mean" without weight
# dfs.select(Summarizer.mean(df.features)).show(truncate=False)
