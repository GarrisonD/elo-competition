# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime
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

# %run ../0-utils/0-Base.py

import os

# Load all parts after feature engineering with Spark and concat into one `DataFrame`:

# +
# %%time
 
SPARK_PARTS_DIR_PATH = "../data/1-feature-engineered"
 
spark_parts_files_names = sorted(os.listdir(SPARK_PARTS_DIR_PATH))
spark_parts_files_names = filter(lambda file_name: file_name.startswith("part-"), spark_parts_files_names)
spark_parts_files_pathes = map(lambda file_name: f"{SPARK_PARTS_DIR_PATH}/{file_name}", spark_parts_files_names)
spark_parts_dfs = list(map(pd.read_csv, spark_parts_files_pathes))
transactions_df = pd.concat(spark_parts_dfs)

display(transactions_df)
# -

# Split transactions info specific number of partitions and save them:

# +
# %%time

for part, part_df in add_part(transactions_df).groupby("part"):
    part_file_path = "../data/2-repartitioned/part-%03d.csv" % part
    mode, header = ("w", True) if not os.path.isfile(part_file_path) else ("a", False)
    part_df.drop("part", axis=1).to_csv(part_file_path, mode=mode, header=header, index=False)
