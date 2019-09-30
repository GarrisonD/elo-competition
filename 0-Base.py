# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime
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

# Import `numpy`:

import numpy as np

# Import `pandas` and configure it:

# +
import pandas as pd

pd.options.display.max_rows = 10
# -

# Import `matplotlib` and configure it:

# +
import matplotlib.pyplot as plt

# %matplotlib inline

# %config InlineBackend.figure_format = 'retina'
# -

# Import `seaborn`:

import seaborn as sns

# Import `tqdm`:

from tqdm.auto import tqdm

# Define global variables:

DATA_PATH = 'data'
