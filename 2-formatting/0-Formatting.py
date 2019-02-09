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

# %run ../0-utils/0-Base.ipynb

# Define a function for formatting a `DataFrame` with transactions:

# +
CAT_COLS_MAPPING = dict(
    authorized_flag=dict(
        Y=1,
        N=0,
    ),
    category_1=dict(
        Y=1,
        N=0,
    ),
    category_3=dict(
        A=0,
        B=1,
        C=2,
    ),
)

CAT_COLS = ("authorized_flag",
            "category_1",
            "category_2",
            "category_3")

def process_transactions_df(df):
    for col, mapping in CAT_COLS_MAPPING.items(): df[col] = df[col].map(mapping)
    for col in CAT_COLS: df[col] = df[col].fillna(-1).astype(np.int)
    return df
# -

# Define a function for processing a specific partition:

# +
import os

from shutil import copyfile

def process_part(part):
    source_part_dir_path = "../data/1-partitioned/%03d" % part
    target_part_dir_path = "../data/2-formatted/%03d"   % part
    
    if not os.path.exists(target_part_dir_path): os.mkdir(target_part_dir_path)
    
    source_transactions_file_path = source_part_dir_path + "/transactions.csv"
    target_transactions_file_path = target_part_dir_path + "/transactions.csv"
    
    transactions_df = pd.read_csv(source_transactions_file_path)
    transactions_df = process_transactions_df(transactions_df)
    transactions_df.to_csv(target_transactions_file_path)
    
    for file_name in ("train.csv", "test.csv"):
        copyfile(source_part_dir_path + "/" + file_name,
                 target_part_dir_path + "/" + file_name)

# process_part(13)
# -

# Perform formatting of transactions:

# +
# %%time

from multiprocessing import Pool

with Pool(8) as p: p.map(process_part, range(TRANSACTIONS_N_PARTS))
