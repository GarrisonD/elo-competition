# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime
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

# +
# %load_ext autoreload

# %autoreload 2
# -

import numpy as np

# +
import pandas as pd

pd.set_option("display.max_columns", 1000)

# +
import matplotlib.pyplot as plt

# %matplotlib inline

# %config InlineBackend.figure_format = "retina"
# -

import seaborn as sns

from tqdm.auto import tqdm

DATA_PATH = "../data"
