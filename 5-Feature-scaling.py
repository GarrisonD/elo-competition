# ---
# jupyter:
#   jupytext:
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

# %run 0-Base.py

# +
NUM_MONTH_LAGS = 16
NUM_FEATURES   = 29

SOURCE_PATH = f'{DATA_PATH}/4-feature-engineered'
TARGET_PATH = f'{DATA_PATH}/5-scaled'
# -

# Read aggregated by *card_id* data of transactions:

# +
# %time aggregated_transactions_df = pd.read_feather(f'{SOURCE_PATH}/aggregated-transactions-by-card-id.feather')

with pd.option_context('display.max_columns', 1000): display(aggregated_transactions_df)
# -

# Define a synthetic feature - *year season* - one of (winter, spring, summer, autumn):

# +
aggregated_transactions_df['season'] = (aggregated_transactions_df['first(purchase_month)'] % 12 + 3) // 3

aggregated_transactions_df.season = aggregated_transactions_df.season.astype('float') # suppress MinMaxScaler warning

# +
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_purchase_amounts(df):
    features = [
        'min(purchase_amount)',
        'mean(purchase_amount)',
        'max(purchase_amount)',

        'min(authorized_purchase_amount)',
        'mean(authorized_purchase_amount)',
        'max(authorized_purchase_amount)',

        'min(not_authorized_purchase_amount)',
        'mean(not_authorized_purchase_amount)',
        'max(not_authorized_purchase_amount)',
    ]

    X = df[features].values

    # get rid of negative values
    X += np.abs(np.nanmin(X, axis=0))
    X += 1e-8 # get rid of zeros
    X = np.log(X)

    X = StandardScaler().fit_transform(X)

    df[features] = X

def process_transactions_count(df):
    X = df[['count']].values

    X = np.log(X)

    X = StandardScaler().fit_transform(X)

    df[['count']] = X

def process_datetime(df):
    features = [
        'first(purchase_month)',
        'first(purchase_year)',
        'season',
    ]

    X = df[features].values

    X = MinMaxScaler().fit_transform(X)

    df[features] = X

# %time process_datetime(aggregated_transactions_df)
# %time process_purchase_amounts(aggregated_transactions_df)
# %time process_transactions_count(aggregated_transactions_df)


# +
_, axs = plt.subplots(nrows=7, ncols=4, figsize=(18, 24))

for ax, feature in zip(axs.flat, aggregated_transactions_df.columns[3:]):
    sns.distplot(aggregated_transactions_df[feature].dropna(), ax=ax)

plt.tight_layout()
# -

# Read data about _train_ and _test_ customers:

# +
train_df = pd.read_csv(f'{DATA_PATH}/raw/train.csv', usecols=['card_id', 'target'])
test_df  = pd.read_csv(f'{DATA_PATH}/raw/test.csv',  usecols=['card_id'])

# %time customers_df = pd.concat([train_df, test_df], sort=False)

# %time customers_df = customers_df.set_index('card_id')

display(customers_df)


# +
def process_customer_transactions_df(df):
    X = -999 * np.ones((NUM_MONTH_LAGS, NUM_FEATURES))
    X[df.month_lag] = df.values[:, 2:]
    X[np.isnan(X)] = -999
    return X

def process_transactions(df):
    df = df.copy()
    df.month_lag += 13 # solve the issue with negative month_lag while indexing

    X, y = [], []

    for card_id, customer_transactions_df in df.groupby('card_id'):
        X.append(process_customer_transactions_df(customer_transactions_df))
        y.append(customers_df.loc[card_id])

    X = np.concatenate(X).reshape(-1, NUM_MONTH_LAGS, NUM_FEATURES)
    y = np.concatenate(y).reshape(-1, 1)

    return X, y

# %time X, y = process_transactions(aggregated_transactions_df)

display(X.shape, y.shape)

# +
train_ixs = np.isfinite(y).ravel()

X_train, X_test, y_train = X[train_ixs], X[~train_ixs], y[train_ixs]

# +
np.save(f'{TARGET_PATH}/train/X.npy', X_train)
np.save(f'{TARGET_PATH}/train/y.npy', y_train)

np.save(f'{TARGET_PATH}/test/X.npy', X_test)
