# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run ../0-utils/0-Base.py

# Define some global variables and helpers:

# +
SOURCE_PATH_PREFIX = "../data/3-feature-engineered/production/train"
TARGET_PATH_PREFIX = "../data/4-features-combined/train"

def read_feature_set(feature_set):
    return np.load(f"{SOURCE_PATH_PREFIX}/{feature_set}.npy")

def read_feature_sets(*feature_sets):
    return np.concatenate(list(map(read_feature_set, feature_sets)), axis=2)


# -


# Read and process __numerical__ feature sets:

# +
from sklearn.preprocessing import StandardScaler

def process_authorized_flag():
    return read_feature_set("authorized_flag")

def process_installments():
    return read_feature_set("installments")

def process_new_transactions():
    return read_feature_set("new_transactions")

def process_purchase_amounts():
    X = read_feature_sets("purchase_amount", "purchase_amount_by_authorized_flag")

    X = X.reshape(-1, 9)
    X += np.abs(np.nanmin(X, axis=0)) + 1e-9
    X[np.isfinite(X)] = np.log(X[np.isfinite(X)])
    X = StandardScaler().fit_transform(X)
    X = X.reshape(-1, 15, 9)
    
    return X

numerical_feature_sets = (
    process_purchase_amounts(),
    process_authorized_flag(),
    process_installments(),
    process_new_transactions(),
)

X_numerical = np.concatenate(numerical_feature_sets, axis=2)

X_numerical.shape
# -

# Read and process feature sets that requires __binning__:

# +
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

def process_transactions_count():
    X = read_feature_set("transactions_count")

    X = X.reshape(-1, 1)
    X_finite_mask = np.isfinite(X)
    X_finite = X[X_finite_mask].reshape(-1, 1)

    # performing binning
    discretizer = KBinsDiscretizer(encode="ordinal")
    X_binned = discretizer.fit_transform(X_finite)
    X[X_finite_mask] = X_binned.ravel()

    # scaling to [0, 1]
    X = MinMaxScaler().fit_transform(X)
    X = X.reshape(-1, 15, 1)
    return X

X_transactions_count = process_transactions_count()

X_transactions_count.shape
# -

X = np.concatenate((X_numerical, X_transactions_count), axis=2)
X[np.isnan(X)] = -1
X.shape

# Save the data for _training_ / _testing_:

# +
from shutil import copyfile

np.save(f"{TARGET_PATH_PREFIX}/X.npy", X)

copyfile(f"{SOURCE_PATH_PREFIX}/y.npy",
         f"{TARGET_PATH_PREFIX}/y.npy");
