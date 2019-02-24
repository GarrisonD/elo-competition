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

# %run 0-utils/0-Base.py

from pyspark.sql.types import *
import pyspark.sql.functions as F

# +
df = spark.read.csv("data/raw/*_transactions.csv", header=True) \
          .withColumn("month_lag", F.col("month_lag").cast(IntegerType())) \
          .withColumn("purchase_amount", F.col("purchase_amount").cast(DoubleType())) \
          .select("card_id", "month_lag", "purchase_amount")

df.printSchema()

# +
# %%time

agg = (F.min("purchase_amount"),
       F.avg("purchase_amount"),
       F.max("purchase_amount"))

feature_set = df.groupBy("card_id", "month_lag").agg(*agg).orderBy("card_id", "month_lag")

feature_set.write.format("csv").option("header", True).save("tmp")
# -

PARTS_DIR_PATH = "data/1-partitioned"
NUM_MONTH_LAGS = 16

df.month_lag += 13

# +
# %%time

from multiprocessing import Pool

def process_customer_transactions_df(df):
    X = -np.ones((NUM_MONTH_LAGS, 3))
    X[df.month_lag] = df.values[:, 2:]
    return X

def process_part(part):
    df = pd.read_csv(f"{PARTS_DIR_PATH}/%03d/transactions.csv" % part)
    df.month_lag += 13
    
    X = list(map(lambda x: process_customer_transactions_df(x[1]), df.groupby("card_id")))
    X = np.concatenate(X).reshape(-1, NUM_MONTH_LAGS, 3)
    return X

with Pool(8) as pool: X = pool.map(process_part, np.arange(TRANSACTIONS_N_PARTS))
# -

np.concatenate(X).shape
