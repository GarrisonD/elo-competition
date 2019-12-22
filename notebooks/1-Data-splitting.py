# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

from elo_competition.utils import split_by_columns

split_by_columns(f'{DATA_PATH}/raw/historical_transactions.csv',
                 f'{DATA_PATH}/1-splitted/historical_transactions')

split_by_columns(f'{DATA_PATH}/raw/new_merchant_transactions.csv',
                 f'{DATA_PATH}/1-splitted/new_merchant_transactions')
