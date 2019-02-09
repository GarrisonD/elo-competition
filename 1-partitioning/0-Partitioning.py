# ---
# jupyter:
#   jupytext:
#     metadata_filter:
#       cells:
#         additional:
#         - ExecuteTime
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

# Perform partitioning of files with transactions:

# +
# %%time

import os

def process_transactions_file(clazz):
    path = f"../data/raw/{clazz}_transactions.csv"
    
    for df in pd.read_csv(path, chunksize=500_000):
        df["new_merchant"] = 1 if "new" == clazz else 0
        
        for part, part_df in add_part(df).groupby("part"):
            part_dir_path = "../data/1-partitioned/%03d" % part
            part_file_path = f"{part_dir_path}/transactions.csv"

            if not os.path.exists(part_dir_path): os.mkdir(part_dir_path)
            mode, header = ("w", True) if not os.path.isfile(part_file_path) else ("a", False)
            part_df.drop("part", axis=1).to_csv(part_file_path, mode=mode, header=header, index=False)

process_transactions_file("old")
process_transactions_file("new")
# -

# Perform partitioning of files with customers:

# +
# %%time

import os

def process_customers_file(clazz):
    df = pd.read_csv(f"../data/raw/{clazz}.csv")
    
    for part, part_df in add_part(df).groupby("part"):
        part_dir_path = "../data/1-partitioned/%03d" % part
        part_file_path = "%s/%s.csv" % (part_dir_path, clazz)
        
        if not os.path.exists(part_dir_path): os.mkdir(part_dir_path)
        part_df.drop("part", axis=1).to_csv(part_file_path, index=False)
        
process_customers_file("train")
process_customers_file("test")
