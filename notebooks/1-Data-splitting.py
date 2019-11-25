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

# %run 0-Base.py

# For example, we have a `test.csv` file with columns `a`, `b`, and `c`.
#
# `split_by_columns` function creates files `a.csv`, `b.csv`, and `c.csv`.

def split_by_columns(source_file_path, target_directory_path):
    files = []

    try:
        source_file = open(source_file_path)
        files.append(source_file)

        columns = source_file.readline().strip().split(',')

        target_files = [open(f'{target_directory_path}/{column}.csv', 'w') for column in columns]
        files.extend(target_files)

        for column, file in zip(columns, target_files):
            file.write(column + '\n')

        for row in tqdm(source_file):
            values = row.strip().split(',')
            assert len(values) == len(columns)

            for value, file in zip(values, target_files):
                if value == '': value = 'NULL'

                file.write(value + '\n')
    finally:
        for file in files:
            file.close()


split_by_columns(f'{DATA_PATH}/raw/historical_transactions.csv',
                 f'{DATA_PATH}/1-splitted/historical_transactions')

split_by_columns(f'{DATA_PATH}/raw/new_merchant_transactions.csv',
                 f'{DATA_PATH}/1-splitted/new_merchant_transactions')
