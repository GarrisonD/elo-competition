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

from elo_competition.data import read_card_owners_csv, reduce_mem_usage

SOURCE_PATH = f"{DATA_PATH}/raw"
TARGET_PATH = f"{DATA_PATH}/1-reformatted"

# #### Reformatting data about card owners:

# +
# %time train_df = read_card_owners_csv(f"{SOURCE_PATH}/train.csv")
# %time train_df = reduce_mem_usage(train_df)

display(train_df)
# -

train_df.info()

# %time train_df.to_feather(f"{TARGET_PATH}/train.feather")

# +
# %time test_df = read_card_owners_csv(f"{SOURCE_PATH}/test.csv")
# %time test_df = reduce_mem_usage(test_df)

display(test_df)
# -

test_df.info()

# %time test_df.to_feather(f"{TARGET_PATH}/test.feather")

# #### Reformatting data about transactions:

from elo_competition.data import read_transactions_csv, reduce_mem_usage

# +
# %time historical_transactions_df = read_transactions_csv(f"{SOURCE_PATH}/historical_transactions.csv")

display(historical_transactions_df)

# +
# %time new_merchant_transactions_df = read_transactions_csv(f"{SOURCE_PATH}/new_merchant_transactions.csv")

display(new_merchant_transactions_df)

# +
# %%time

transactions_df = pd.concat(
    (
        historical_transactions_df,
        new_merchant_transactions_df,
    ),
    ignore_index=True,
    copy=False,
)
# -

# %time transactions_df = reduce_mem_usage(transactions_df)

transactions_df.info()

# %time transactions_df.to_feather(f"{TARGET_PATH}/transactions.feather")
