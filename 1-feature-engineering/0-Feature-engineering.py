# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run ../0-utils/0-Base.py

# +
import pyspark.sql.types as T
import pyspark.sql.functions as F

spark.conf.set("spark.sql.shuffle.partitions", 4)
spark.conf.set("spark.sql.execution.arrow.enabled", True)
# -

# Load the data about historical and eval period transactions:

# +
df = spark.read.csv("../data/raw/*_transactions.csv", header=True)

df = df.select("card_id", "month_lag", "purchase_amount")
df = df.withColumn("month_lag", F.col("month_lag").cast(T.IntegerType()))
df = df.withColumn("purchase_amount", F.col("purchase_amount").cast(T.DoubleType()))

df.printSchema()
# -

# Calculate features and save them:

# +
# %%time

agg = (F.min("purchase_amount"),
       F.avg("purchase_amount"),
       F.max("purchase_amount"))

feature_set = df.groupBy("card_id", "month_lag").agg(*agg).orderBy("card_id", "month_lag")

# feature_set.write.format("csv") \
#            .mode("overwrite") \
#            .option("header", True) \
#            .save("../data/1-feature-engineered")

# display(feature_set.toPandas().to_feather)
dataframe = feature_set.toPandas()
# -

display(dataframe)

dataframe.to_feather("../data/1-feature-engineered/feather.csv")
