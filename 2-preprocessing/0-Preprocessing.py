# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run ../0-utils/0-Base.py

# Define global variables:

NUM_MONTH_LAGS = 16
NUM_FEATURES   = 28

df = pd.read_feather("../data/1-feature-engineered/aggregated-transactions-by-card-id.feather"); display(df)

# +
df["seasons_time"] = 0

df.loc[((1. <= df["avg(purchase_month)"]) & (df["avg(purchase_month)"] <= 2.))   |  (df["avg(purchase_month)"] == 12), "seasons_time"] = 1
df.loc[(3. <= df["avg(purchase_month)"]) & (df["avg(purchase_month)"] <= 5.), "seasons_time"] = 2
df.loc[(6. <= df["avg(purchase_month)"]) & (df["avg(purchase_month)"] <= 8.), "seasons_time"] = 3
df.loc[(9. <= df["avg(purchase_month)"]) & (df["avg(purchase_month)"] <= 11.) , "seasons_time"] = 4
df.session_time.unique()

# +
# %%time

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_purchase_amounts(df):
    features = [
        "min(purchase_amount)",
        "avg(purchase_amount)",
        "max(purchase_amount)",
        
        "min(authorized_purchase_amount)",
        "avg(authorized_purchase_amount)",
        "max(authorized_purchase_amount)",
        
        "min(not_authorized_purchase_amount)",
        "avg(not_authorized_purchase_amount)",
        "max(not_authorized_purchase_amount)",
    ]
    
    X = df[features].values

    # get rid of negative values
    X += np.abs(np.nanmin(X, axis=0))
    X += 1e-8 # get rid of zeros
    X = np.log(X)
    
    X = StandardScaler().fit_transform(X)
    
    df[features] = X
    
def process_transactions_count(df):
    X = df[["count"]].values
    X = np.log(X)
    
    X = StandardScaler().fit_transform(X)
    
    df[["count"]] = X
    
def process_datetime(df):
    features = [
        "avg(purchase_year)",
        "avg(purchase_month)",
        "seasons_time",
    ]
    
    X = df[features].values

    X = MinMaxScaler().fit_transform(X)
    
    df[features] = X
    
process_purchase_amounts(df)
process_transactions_count(df)
process_datetime(df)

# +
import matplotlib.pyplot as plt
import numpy as np

methods = np.array(df.columns[3:])

fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(18, 18))

for ax, interp_method in zip(axs.flat, methods):
    sns.distplot(df[interp_method],bins=50,ax=ax)
plt.tight_layout()
plt.show()
# -

# Read data about _train_ and _test_ customers:

# +
train_customers_df = pd.read_csv("../data/raw/train.csv", usecols=("card_id", "target"))
test_customers_df  = pd.read_csv("../data/raw/test.csv",  usecols=("card_id",))
customers_df = pd.concat((train_customers_df, test_customers_df), sort=False)
customers_df = customers_df.set_index("card_id")

display(customers_df)


# +
# %%time

def process_customer_transactions_df(df):
    X = -999 * np.ones((NUM_MONTH_LAGS, NUM_FEATURES))
    X[df.month_lag] = df.values[:, 2:]
    X[np.isnan(X)] = -999
    return X

def process_transactions(df):
    df = df.copy()
    df.month_lag += 13 # solve the issue with negative month_lag while indexing

    X, y = [], []
    
    for card_id, customer_transactions_df in df.groupby("card_id"):
        X.append(process_customer_transactions_df(customer_transactions_df))
        y.append(customers_df.loc[card_id])

    X = np.concatenate(X).reshape(-1, NUM_MONTH_LAGS, NUM_FEATURES)
    y = np.concatenate(y).reshape(-1, 1)

    return X, y

X, y = process_transactions(df)

display(X.shape, y.shape)

# +
train_ixs = np.isfinite(y).ravel()
test_ixs  = np.isnan(y).ravel()

X_train, X_test, y_train = X[train_ixs], X[test_ixs], y[train_ixs]

# +
np.save("../data/3-preprocessed/train/X.npy", X_train)
np.save("../data/3-preprocessed/train/y.npy", y_train)

np.save("../data/3-preprocessed/test/X.npy", X_test)
