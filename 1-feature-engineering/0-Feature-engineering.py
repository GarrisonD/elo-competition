# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
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

# +
# %%time

df = spark.read.csv("../data/raw/*_transactions.csv", header=True)
df = df.withColumn("month_lag", F.col("month_lag").cast(T.IntegerType()))
df = df.withColumn("purchase_date", F.col("purchase_date").cast(T.DateType()))
df = df.withColumn("purchase_amount", F.col("purchase_amount").cast(T.DoubleType()))

df = df.withColumn("authorized_flag", F.when(df.authorized_flag == "Y", 1) \
                                       .when(df.authorized_flag == "N", 0))

for key, value in dict(authorized_purchase_amount=1, not_authorized_purchase_amount=0).items():
    df = df.withColumn(key, F.when(df.authorized_flag == value, df.purchase_amount).otherwise(None))
    
df = df.withColumn("installments", F.when(df.installments != "999", df.installments).otherwise("-1"))

for value in range(-1, 13):
    df = df.withColumn(f"installments_{value}", F.when(df.installments == value, 1).otherwise(0))
    
df = df.withColumn("purchase_year", F.year("purchase_date"))
df = df.withColumn("purchase_month", F.month("purchase_date"))

agg = (
    F.avg("authorized_flag"),
    
    F.min("purchase_amount"),
    F.avg("purchase_amount"),
    F.max("purchase_amount"),
    
    F.count(F.lit(1)).alias("count"),
    
    F.first("purchase_year", True),
    F.first("purchase_month", True),
    
    F.min("authorized_purchase_amount"),
    F.avg("authorized_purchase_amount"),
    F.max("authorized_purchase_amount"),
    
    F.min("not_authorized_purchase_amount"),
    F.avg("not_authorized_purchase_amount"),
    F.max("not_authorized_purchase_amount"),
    
    F.avg("installments_-1"),
    F.avg("installments_0"),
    F.avg("installments_1"),
    F.avg("installments_2"),
    F.avg("installments_3"),
    F.avg("installments_4"),
    F.avg("installments_5"),
    F.avg("installments_6"),
    F.avg("installments_7"),
    F.avg("installments_8"),
    F.avg("installments_9"),
    F.avg("installments_10"),
    F.avg("installments_11"),
    F.avg("installments_12"),
)

pandas_df = df.groupBy("card_id", "month_lag").agg(*agg).orderBy("card_id", "month_lag").toPandas()

display(pandas_df)
# -

# Calculate features and save them:

# +
# %%time

pandas_df.to_feather("../data/1-feature-engineered/aggregated-transactions-by-card-id.feather")

# +
# %%time

from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.stat import Summarizer

df = spark.read.csv("../data/raw/*_transactions.csv", header=True)
df.printSchema()

# Map authorized flag
dicts = sc.broadcast(dict([('Y', 1), ('N', 0)]))

def newCols(x): return dicts.value[x]
callnewColsUdf = F.udf(newCols, T.IntegerType())
df = df.withColumn("authorized_flag_indexed", callnewColsUdf(F.col("authorized_flag")))

# map installments to int, fix -1 value
callNewColsUdf = F.udf(lambda x: 999 if x == "-1" else int(x), T.IntegerType())
df = df.withColumn("installments_fixed", callNewColsUdf(F.col("installments")))

# Produce vector for installments
# dropLast=True set by default. That produce value 999 to vector of zeros [0]. 
encoder = OneHotEncoderEstimator(inputCols=["installments_fixed"],
                                 outputCols=["installments_vec"])
model = encoder.fit(df)
df = model.transform(df)
df.printSchema()

df = df.select("card_id", "month_lag", "purchase_amount", "installments_vec", "authorized_flag_indexed")
df = df.withColumn("month_lag", F.col("month_lag").cast(T.IntegerType()))
df = df.withColumn("purchase_amount", F.col("purchase_amount").cast(T.DoubleType()))
df = df.withColumn("authorized_flag_indexed", F.col("authorized_flag_indexed").cast(T.IntegerType()))

agg = (F.min("purchase_amount"),
       F.avg("purchase_amount"),
       F.max("purchase_amount"),
       F.count("purchase_amount"),
       F.avg("authorized_flag_indexed"),
       Summarizer.mean(df.installments_vec))

feature_set = df.groupBy("card_id", "month_lag").agg(*agg).orderBy("card_id", "month_lag")

feature_set.show()
