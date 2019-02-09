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

# Load an extension for profiling:

# %load_ext line_profiler

# Import `numpy`:

import numpy as np

# Import `pandas` and configure it:

# +
import pandas as pd
pd.options.display.max_rows = 6
pd.options.display.max_columns = 0

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# -

# Import `matplotlib` and configure it:

# +
import matplotlib.pyplot as plt

# %matplotlib inline

# %config InlineBackend.figure_format = "retina"
# -

# Define global variables:

TRANSACTIONS_N_PARTS = 256

# Define global helpers:

# +
from hashlib import md5

def to_md5(x): return md5(x.encode("utf-8"))
def to_md5_int(x): return int(to_md5(x).hexdigest(), 16)
def card_id_to_part(x): return to_md5_int(x) % TRANSACTIONS_N_PARTS
# -

def add_part(df): return df.assign(part=df.card_id.map(card_id_to_part))
