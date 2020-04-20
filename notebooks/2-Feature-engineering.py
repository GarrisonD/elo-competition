# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run 0-Base.ipynb

SOURCE_PATH = f"{DATA_PATH}/1-reformatted"
TARGET_PATH = f"{DATA_PATH}/2-feature-engineered"

# %time transactions_df = pd.read_feather(f"{SOURCE_PATH}/transactions.feather")

# %time transactions_df[    "authorized_purchase_amount"] = transactions_df.purchase_amount.where( transactions_df.authorized_flag)
# %time transactions_df["not_authorized_purchase_amount"] = transactions_df.purchase_amount.where(~transactions_df.authorized_flag)

# +
# %%time

transactions_df["purchase_year" ] = transactions_df.purchase_date.dt.year
transactions_df["purchase_month"] = transactions_df.purchase_date.dt.month
# -

# %time transactions_df = pd.get_dummies(transactions_df, columns=["installments"], dtype="int8")

agg = {
    "authorized_flag": ["count", "mean"],

    "purchase_year":  ["first"],
    "purchase_month": ["first"],

    "purchase_amount":                ["min", "mean", "max"],
    "authorized_purchase_amount":     ["min", "mean", "max"],
    "not_authorized_purchase_amount": ["min", "mean", "max"],

    "installments_-1":  ["mean"],
    "installments_0":   ["mean"],
    "installments_1":   ["mean"],
    "installments_2":   ["mean"],
    "installments_3":   ["mean"],
    "installments_4":   ["mean"],
    "installments_5":   ["mean"],
    "installments_6":   ["mean"],
    "installments_7":   ["mean"],
    "installments_8":   ["mean"],
    "installments_9":   ["mean"],
    "installments_10":  ["mean"],
    "installments_11":  ["mean"],
    "installments_12":  ["mean"],
    "installments_999": ["mean"],
}

# %time aggregated_transactions_df = transactions_df.groupby(["card_id", "month_lag"]).agg(agg)
aggregated_transactions_df.columns = [f"{col}_{fn}" for col, fn in aggregated_transactions_df.columns]
aggregated_transactions_df = aggregated_transactions_df.rename(columns={"authorized_flag_count": "count"}).reset_index()

# Define a synthetic feature - *season of year* - one of (winter, spring, summer, autumn):

# %time aggregated_transactions_df["season"] = (aggregated_transactions_df["purchase_month_first"] % 12 + 3) // 3

aggregated_transactions_df.info()

# +
from elo_competition.data import reduce_mem_usage

# %time aggregated_transactions_df = reduce_mem_usage(aggregated_transactions_df)
# -

aggregated_transactions_df.info()

# %time aggregated_transactions_df.to_feather(f"{TARGET_PATH}/aggregated-transactions-by-card-id.feather")
