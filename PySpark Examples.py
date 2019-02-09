# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run 0-utils/0-Base.ipynb

from pyspark.sql.types import *
from pyspark.sql.functions import *

# +
# %%time

# schema = StructType([
#     StructField("authorized_flag",      ShortType(),     nullable=False),
#     StructField("card_id",              StringType(),    nullable=False),
#     StructField("city_id",              ShortType(),     nullable=False),
#     StructField("category_1",           ShortType(),     nullable=False),
#     StructField("installments",         ShortType(),     nullable=False),
#     StructField("category_3",           StringType(),    nullable=True),
#     StructField("merchant_category_id", ShortType(),     nullable=False),
#     StructField("merchant_id",          StringType(),    nullable=True),
#     StructField("month_lag",            ShortType(),     nullable=False),
#     StructField("purchase_amount",      FloatType(),     nullable=False),
#     StructField("purchase_date",        TimestampType(), nullable=False),
#     StructField("category_2",           FloatType(),     nullable=True),
#     StructField("state_id",             ShortType(),     nullable=False),
#     StructField("subsector_id",         ShortType(),     nullable=False),
# ])

transactions_dfs = dict(old=spark.read.csv("data/raw/old_transactions.csv", header=True),
                        new=spark.read.csv("data/raw/new_transactions.csv", header=True))

# +
# %%time

for transactions_clazz, transactions_df in transactions_dfs.items():
    transactions_df = transactions_df.where(transactions_df.card_id == "C_ID_3d0044924f")
    transactions_df = transactions_df.withColumn("purchase_date", col("purchase_date").cast("date"))

    print(transactions_clazz.upper())
    
    transactions_df.select(
        min("purchase_date"),
        max("purchase_date"),
    ).show()
    
    purchase_dates = transactions_df.select("purchase_date").collect()
    purchase_dates = list(map(lambda x: x.purchase_date, purchase_dates))
    purchase_dates = pd.to_datetime(purchase_dates)

    fig = plt.figure()
    plt.hist(purchase_dates)
    fig.autofmt_xdate()
    plt.show()
# -

spark.read.csv("data/raw/train.csv", header=True).where("target < -33.").show()

# +
# %%time

tmp_df = spark.read.csv("data/1-partitioned/*/transactions.csv", header=True)

tmp_df = tmp_df.where(tmp_df.card_id == "C_ID_962d44f420")
tmp_df = tmp_df.withColumn("authorized_flag", when(tmp_df.authorized_flag == 'Y', 1).otherwise(0))
tmp_df = tmp_df.withColumn("purchase_date", col("purchase_date").cast("timestamp"))

tmp_df = tmp_df.groupBy(
    year("purchase_date"),
    month("purchase_date"),
)

tmp_df = tmp_df.agg(
    count("authorized_flag"),
    mean("authorized_flag"),
)

tmp_df.sort(
    "year(purchase_date)",
    "month(purchase_date)",
).show()

# +
tmp_df = spark.read.csv("data/1-partitioned/*/transactions.csv", header=True)

tmp_df.groupBy("card_id").count().select(min("count"), max("count")).show()

# +
tmp_df = spark.read.csv("data/1-partitioned/*/transactions.csv", header=True)
tmp_df = tmp_df.withColumn("purchase_date", col("purchase_date").cast("date"))

tmp_df.select(
    min("purchase_date"),
    max("purchase_date"),
).show()

# +
tmp_df = spark.read.csv("data/1-partitioned/**/transactions.csv", header=True)

tmp_df = tmp_df.withColumn("purchase_amount", col("purchase_amount").cast("float"))

tmp_df.select(
    # as is
    min("purchase_amount"),
    max("purchase_amount"),
    
    # with log
    min(log(col("purchase_amount") + 1)),
    max(log(col("purchase_amount") + 1)),
).show()
