# ---
# jupyter:
#   jupytext:
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

# Define global variables:

NUM_MONTH_LAGS = 16

# +
feather_df = pd.read_feather("../data/1-feature-engineered/feather.csv")

display(feather_df)

# +
from sklearn.preprocessing import StandardScaler

def process_purchase_amounts(X):
    X += np.abs(np.min(X, axis=0)) + 1e-9
    X = np.log(X)
    X = StandardScaler().fit_transform(X)

    return X
    
scaled_df = process_purchase_amounts(feather_df[feather_df.columns[2:]].values)
# -

sns.distplot(scaled_df[:, 1], bins=50)

feather_df[feather_df.columns[2:]] = scaled_df

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
    X = -np.ones((NUM_MONTH_LAGS, 3))
    X[df.month_lag] = df.values[:, 2:]
    return X

def process_transactions(df):
    df.month_lag += 13 # solve the issue with negative month_lag while indexing

    X, y = [], []
    
    for card_id, customer_transactions_df in df.groupby("card_id"):
        X.append(process_customer_transactions_df(customer_transactions_df))
        y.append(customers_df.loc[card_id])

    X = np.concatenate(X).reshape(-1, NUM_MONTH_LAGS, 3)
    y = np.concatenate(y).reshape(-1, 1)

    return X, y

results = process_transactions(feather_df)

# +
# X = list(x[0] for x in results)
# y = list(x[1] for x in results)

# X = np.concatenate(X)
# y = np.concatenate(y)

X, y = results

display(X.shape, y.shape)

# +
train_ixs = np.isfinite(y).ravel()
test_ixs  = np.isnan(y).ravel()

X_train, X_test, y_train = X[train_ixs], X[test_ixs], y[train_ixs]

# +
np.save("../data/3-preprocessed/train/X.npy", X_train)
np.save("../data/3-preprocessed/train/y.npy", y_train)

np.save("../data/3-preprocessed/test/X.npy", X_test)

# +
# %%time

from multiprocessing import Pool

def process_customer_transactions_df(df):
    X = -np.ones((NUM_MONTH_LAGS, 3))
    X[df.month_lag] = df.values[:, 2:]
    return X

def process_transactions_part(part):
    df = pd.read_csv(f"../data/2-repartitioned/part-%03d.csv" % part)
    df.month_lag += 13 # solve the issue with negative month_lag while indexing

    X, y = [], []
    
    for card_id, customer_transactions_df in df.groupby("card_id"):
        X.append(process_customer_transactions_df(customer_transactions_df))
        y.append(customers_df.loc[card_id])

    X = np.concatenate(X).reshape(-1, NUM_MONTH_LAGS, 3)
    y = np.concatenate(y).reshape(-1, 1)
    
    return X, y

with Pool(8) as pool: results = pool.map(process_transactions_part, np.arange(TRANSACTIONS_N_PARTS))
