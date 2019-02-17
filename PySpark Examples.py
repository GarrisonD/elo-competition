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

# %run 0-utils/0-Base.py

from pyspark.sql.types import *
from pyspark.sql.functions import *

tmp_df = spark.read.csv("data/1-partitioned/**/transactions.csv", header=True)

tmp_df. \
    where(tmp_df.card_id == "C_ID_8186f3fcc1"). \
    withColumn("purchase_date", tmp_df.purchase_date.cast("date")). \
    groupBy(["card_id", "new_merchant"]).agg(min("purchase_date"), max("purchase_date")). \
    show()

# +
customers_df = spark.read.csv("data/raw/train.csv", header=True)

customers_df.where(customers_df.target < -33.).show()
