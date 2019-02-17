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

# Define some global variables and helpers:

# +
SOURCE_PATH_PREFIX = "../data/3-feature-engineered/train"
TARGET_PATH_PREFIX = "../data/4-features-combined/train"

def read_feature_set(feature_set):
    return np.load(f"{SOURCE_PATH_PREFIX}/{feature_set}.npy")

def read_feature_sets(*feature_sets):
    return list(map(read_feature_set, feature_sets))
# -

# Read and process __numerical__ feature sets:

# +
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def process_authorized_flag():
    X = read_feature_set("authorized_flag")

    X = X.reshape(-1, 1)
    X = MinMaxScaler().fit_transform(X)
    X = X.reshape(-1, 15, 1)

    return X

def process_purchase_amounts():
    feature_sets = read_feature_sets(
        "purchase_amount",
        "purchase_amount_by_authorized_flag",
    )

    X = np.concatenate(feature_sets, axis=2)

    X = X.reshape(-1, 9)
    X += np.abs(np.nanmin(X, axis=0)) + 1e-9
    X[np.isfinite(X)] = np.log(X[np.isfinite(X)])
    X = StandardScaler().fit_transform(X)
    X = X.reshape(-1, 15, 9)
    
    return X

numerical_feature_sets = (process_authorized_flag(),
                          process_purchase_amounts())

X_numerical = np.concatenate(numerical_feature_sets, axis=2)

# # missing values imputation
X_numerical[np.isnan(X_numerical)] = -1

X_numerical.shape
# -

# Read and process feature sets that requires __binning__:

# +
from sklearn.preprocessing import KBinsDiscretizer

def process_transactions_count():
    X = read_feature_set("transactions_count")

    X = X.reshape(-1, 1)
    X_finite_mask = np.isfinite(X)
    X_finite = X[X_finite_mask].reshape(-1, 1)

    X_finite[X_finite > 50] = 50
    sns.distplot(X_finite.ravel())

    discretizer = KBinsDiscretizer(strategy="quantile")
    X_binned = discretizer.fit_transform(X_finite)

    display(discretizer.bin_edges_)

    X = -np.ones((X.shape[0] * X.shape[1], X_binned.shape[1]))
    X[X_finite_mask.ravel()] = X_binned.toarray()
    X = X.reshape(-1, 15, 5)

    return X

X_transactions_count = process_transactions_count()

X_transactions_count.shape

# +
X = np.concatenate((X_numerical, X_transactions_count), axis=2)

X.shape
# -

# Save the data for _training_ / _testing_:

# +
from shutil import copyfile

np.save(f"{TARGET_PATH_PREFIX}/X.npy", X)

copyfile(f"{SOURCE_PATH_PREFIX}/y.npy",
         f"{TARGET_PATH_PREFIX}/y.npy");
