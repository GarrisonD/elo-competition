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

# %run ../0-utils/0-Base.py

# Define some global variables:

SOURCE_PATH_PREFIX = "../data/3-feature-engineered/train"
TARGET_PATH_PREFIX = "../data/4-features-combined/train"

# Read a list of all available feature sets:

# +
import os

feature_sets = []

for file_name in sorted(os.listdir(SOURCE_PATH_PREFIX)):
    if file_name.startswith(".") or file_name == "y.npy": continue
    
    print("Loading %s..." % file_name)
    
    feature_set_file_path = f"{SOURCE_PATH_PREFIX}/{file_name}"
    feature_set = np.load(feature_set_file_path)
    feature_sets.append(feature_set)

print("Loaded %d feature sets!" % len(feature_sets))
# -

# Combine feature sets into one `numpy.ndarray`:

X = np.concatenate(feature_sets, axis=2); X.shape

# Perform feature scaling and missing values imputation:

# +
from sklearn.preprocessing import MinMaxScaler

X = X.reshape(-1, 10)

# feature scaling
X = MinMaxScaler().fit_transform(X)

# missing values imputation
X[np.isnan(X)] = -1

X = X.reshape(-1, 15, 10)
# -

# Save the data for _training_ / _testing_:

# +
from shutil import copyfile

np.save(f"{TARGET_PATH_PREFIX}/X.npy", X)

copyfile(f"{SOURCE_PATH_PREFIX}/y.npy",
         f"{TARGET_PATH_PREFIX}/y.npy");
