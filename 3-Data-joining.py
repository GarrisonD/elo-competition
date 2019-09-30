# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run 0-Base.py

SOURCE_PATH = f'{DATA_PATH}/2-reformatted'
TARGET_PATH = f'{DATA_PATH}/3-joined'

# +
from glob import glob

def join_reformatted_columns(source_directory_path):
    source_file_paths = sorted(glob(f'{source_directory_path}/*.feather'))
    source_dfs = [pd.read_feather(source_file_path) for source_file_path in source_file_paths]
    
    # check that all the data-frames have the same number of rows
    assert len(set(map(lambda df: df.shape[0], source_dfs))) == 1
    
    return pd.concat(source_dfs, axis=1)


# -

# ***

historical_transactions = join_reformatted_columns(f'{SOURCE_PATH}/historical_transactions')
assert len(historical_transactions.columns) == (14 + 3)

historical_transactions.info()

# %time historical_transactions.to_feather(f'{TARGET_PATH}/historical_transactions.feather')

# ***

new_merchant_transactions = join_reformatted_columns(f'{SOURCE_PATH}/new_merchant_transactions')
assert len(new_merchant_transactions.columns) == (14 + 3)

new_merchant_transactions.info()

# %time new_merchant_transactions.to_feather(f'{TARGET_PATH}/new_merchant_transactions.feather')
