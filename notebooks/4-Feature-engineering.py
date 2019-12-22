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

SOURCE_PATH = f'{DATA_PATH}/3-joined'
TARGET_PATH = f'{DATA_PATH}/4-feature-engineered'

# %time historical_transactions_df   = pd.read_feather(f'{SOURCE_PATH}/historical_transactions.feather')
# %time new_merchant_transactions_df = pd.read_feather(f'{SOURCE_PATH}/new_merchant_transactions.feather')

# +
# %time transactions_df = pd.concat([historical_transactions_df, new_merchant_transactions_df])

del historical_transactions_df
del new_merchant_transactions_df
# -

# %time transactions_df[    'authorized_purchase_amount'] = transactions_df.purchase_amount.where( transactions_df.authorized_flag)
# %time transactions_df['not_authorized_purchase_amount'] = transactions_df.purchase_amount.where(~transactions_df.authorized_flag)

# +
# %%time

transactions_df['purchase_year' ] = transactions_df.purchase_date.dt.year
transactions_df['purchase_month'] = transactions_df.purchase_date.dt.month
# -

# %time transactions_df = pd.get_dummies(transactions_df, columns=['installments'], dtype='int8')

agg = {
    'authorized_flag': ['count', 'mean'],

    'purchase_year':  ['first'],
    'purchase_month': ['first'],

    'purchase_amount':                ['min', 'mean', 'max'],
    'authorized_purchase_amount':     ['min', 'mean', 'max'],
    'not_authorized_purchase_amount': ['min', 'mean', 'max'],

    'installments_-1':  ['mean'],
    'installments_0':   ['mean'],
    'installments_1':   ['mean'],
    'installments_2':   ['mean'],
    'installments_3':   ['mean'],
    'installments_4':   ['mean'],
    'installments_5':   ['mean'],
    'installments_6':   ['mean'],
    'installments_7':   ['mean'],
    'installments_8':   ['mean'],
    'installments_9':   ['mean'],
    'installments_10':  ['mean'],
    'installments_11':  ['mean'],
    'installments_12':  ['mean'],
    'installments_999': ['mean'],
}

# +
# %time aggregated_transactions_df = transactions_df.groupby(['card_id', 'month_lag']).agg(agg)
aggregated_transactions_df.columns = [f'{fn}({col})' for col, fn in aggregated_transactions_df.columns]
aggregated_transactions_df = aggregated_transactions_df.rename(columns={'count(authorized_flag)': 'count'}).reset_index()

with pd.option_context('display.max_columns', 1000): display(aggregated_transactions_df)
# -

# Define a synthetic feature - *season of year* - one of (winter, spring, summer, autumn):

# %time aggregated_transactions_df['season'] = (aggregated_transactions_df['first(purchase_month)'] % 12 + 3) // 3

# %time aggregated_transactions_df.to_feather(f'{TARGET_PATH}/aggregated-transactions-by-card-id.feather')
