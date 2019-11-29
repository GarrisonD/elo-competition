import os, pytest
import pandas as pd
from numpy import nan

from elo_competition.utils import split_by_columns

SOURCE_DIR_PATH = 'tests/split-by-columns-source'
TARGET_DIR_PATH = 'tests/split-by-columns-target'

def cleanup_target_dir():
    os.system(f'rm -rf {TARGET_DIR_PATH}/*')

setup_module    = cleanup_target_dir
teardown_module = cleanup_target_dir

def test_split_by_columns():
    split_by_columns(f'{SOURCE_DIR_PATH}/data.csv', TARGET_DIR_PATH)

    A_df = pd.read_csv(f'{TARGET_DIR_PATH}/A.csv')
    assert A_df.A.tolist() == [1, 2, 3]
    assert A_df.dtypes.A == 'int64'
    assert A_df.columns == ['A']

    B_df = pd.read_csv(f'{TARGET_DIR_PATH}/B.csv')
    assert B_df.B.tolist() == [nan, 'B', 'B']
    assert B_df.dtypes.B == 'object'
    assert B_df.columns == ['B']

    C_df = pd.read_csv(f'{TARGET_DIR_PATH}/C.csv')
    assert C_df.C.tolist() == [True, False, False]
    assert C_df.dtypes.C == 'bool'
    assert C_df.columns == ['C']
