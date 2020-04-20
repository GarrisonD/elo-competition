# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# %run 0-Base.ipynb

# +
NUM_MONTH_LAGS = 16
NUM_FEATURES   = 35

SOURCE_PATH = f"{DATA_PATH}/2-feature-engineered"
TARGET_PATH = f"{DATA_PATH}/3-scaled"
# -

# %time aggregated_transactions_df = pd.read_feather(f"{SOURCE_PATH}/aggregated-transactions-by-card-id.feather")

aggregated_transactions_df.isnull().sum()

# +
# %%time

columns = [
    "authorized_purchase_amount_min",
    "authorized_purchase_amount_mean",
    "authorized_purchase_amount_max",
    "not_authorized_purchase_amount_min",
    "not_authorized_purchase_amount_mean",
    "not_authorized_purchase_amount_max",
]

for column in columns:
    aggregated_transactions_df[f"{column}_missing"] = \
        aggregated_transactions_df[column].isnull()
    
    aggregated_transactions_df[column].fillna(
        aggregated_transactions_df[column].mean(),
        inplace=True
    )
# -

# %time aggregated_transactions_df = aggregated_transactions_df[sorted(aggregated_transactions_df.columns)]

aggregated_transactions_df.isnull().sum()

scaled_features = set()

# +
# %%time

from sklearn.preprocessing import StandardScaler

features = [
    "purchase_amount_min",
    "purchase_amount_mean",
    "purchase_amount_max",
    
    "authorized_purchase_amount_min",
    "authorized_purchase_amount_mean",
    "authorized_purchase_amount_max",

    "not_authorized_purchase_amount_min",
    "not_authorized_purchase_amount_mean",
    "not_authorized_purchase_amount_max",
]

df = aggregated_transactions_df[features]

df += np.abs(df.min())
df = np.log(df, where=df!=0)
df = StandardScaler().fit_transform(df)
aggregated_transactions_df[features] = df

scaled_features |= set(features)

# +
# %%time

from sklearn.preprocessing import StandardScaler

features = ["count"]

df = aggregated_transactions_df[features]

df = np.log(df)
df = StandardScaler().fit_transform(df)
aggregated_transactions_df[features] = df

scaled_features |= set(features)

# +
# %%time

from sklearn.preprocessing import MinMaxScaler

features = [
    "purchase_month_first",
    "purchase_year_first",
    "season",
]

df = aggregated_transactions_df[features]
df = MinMaxScaler((-1, 1)).fit_transform(df)
aggregated_transactions_df[features] = df

scaled_features |= set(features)
# -

set(aggregated_transactions_df.columns) - scaled_features

aggregated_transactions_df.info()

# +
non_visualizable_features = {
    "authorized_purchase_amount_max_missing",
    "authorized_purchase_amount_mean_missing",
    "authorized_purchase_amount_min_missing",
    
    "card_id",
    
    "not_authorized_purchase_amount_max_missing",
    "not_authorized_purchase_amount_mean_missing",
    "not_authorized_purchase_amount_min_missing",
}

visualizable_features = set(aggregated_transactions_df.columns) - non_visualizable_features

for feature in sorted(visualizable_features):
    sns.distplot(aggregated_transactions_df[feature])
    
    plt.show()

# +
train_df = pd.read_feather(f"{SOURCE_PATH}/train.feather")
test_df = pd.read_feather(f"{SOURCE_PATH}/test.feather")

card_owners_df = pd.concat(
    [
        train_df,
        test_df,
    ],
    ignore_index=True,
    copy=False,
    sort=False,
)

# %time card_owners_df.set_index("card_id", inplace=True)

display(card_owners_df)


# +
def process_customer_transactions_df(df):
    X = -10 * np.ones((NUM_MONTH_LAGS, NUM_FEATURES))
    X[df.month_lag + 13] = df.drop(columns=["card_id", "month_lag"]).values
    X[np.isnan(X)] = -10
    return X

def process_transactions(df):
    X, y = [], []

    for card_id, customer_transactions_df in df.groupby("card_id"):
        X.append(process_customer_transactions_df(customer_transactions_df))
        y.append(card_owners_df.loc[card_id])

    X = np.concatenate(X).reshape(-1, NUM_MONTH_LAGS, NUM_FEATURES)
    y = np.concatenate(y).reshape(-1, 1)

    return X, y

# %time X, y = process_transactions(aggregated_transactions_df)

display(X.shape, y.shape)

# +
train_ixs = np.isfinite(y).ravel()

X_train, X_test, y_train = X[train_ixs], X[~train_ixs], y[train_ixs]

# +
np.save(f"{TARGET_PATH}/train/X.npy", X_train)
np.save(f"{TARGET_PATH}/train/y.npy", y_train)

np.save(f"{TARGET_PATH}/test/X.npy", X_test)
