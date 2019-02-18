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

# Define helpers for reading data about _customers_ and their _transactions_:

# +
from pandas.api.types import CategoricalDtype

PARTS_DIR_PATH = "../data/2-formatted"

def read_part_customers_df(part, clazz):
    return pd.read_csv(f"{PARTS_DIR_PATH}/%03d/{clazz}.csv" % part)

display(read_part_customers_df(13, "test"),
        read_part_customers_df(13, "train"))

def read_part_transactions_df(part):
    df = pd.read_csv(f"{PARTS_DIR_PATH}/%03d/transactions.csv" % part, parse_dates=["purchase_date"])
    
    # merge 999 and -1 because both of them represent missing values
    df.loc[df.installments == 999, "installments"] = -1
    
    df.installments = df.installments.astype(
        CategoricalDtype(
            categories=np.arange(-1, 13),
            ordered=True,
        )
    )
    
    return df

display(read_part_transactions_df(13))
# -

# Define months for which transactions data will be collected:

dates = pd.date_range(start="2017-01", end="2018-04", freq="M").to_period("M"); display(dates)

# +
development = True
feature_set = "installments"

def process_part(part, clazz):
    part_customers_df = read_part_customers_df(part, clazz)
    max_num_customers = int(part_customers_df.shape[0] / 5)
    
    part_transactions_df = read_part_transactions_df(part)
    
    part_transactions_df = part_transactions_df.assign(
        year=part_transactions_df.purchase_date.dt.year,
        month=part_transactions_df.purchase_date.dt.month,
    )
    
    part_transactions_df = part_transactions_df. \
        set_index(["card_id", "year", "month"]). \
        sort_index()
    
    X, y = [], []
        
    for customer in part_customers_df.itertuples(index=False):
        if development and clazz == "train" and max_num_customers < 0 and customer.target > -33: continue
        
        X_parts = []
        
        for date in dates:
            ix = (customer.card_id, date.year, date.month)
            
            if not part_transactions_df.index.contains(ix):
                if feature_set == "authorized_flag":
                    num_features = 1

                if feature_set == "purchase_amount":
                    num_features = 3

                if feature_set == "purchase_amount_by_authorized_flag":
                    num_features = 6

                if feature_set == "transactions_count":
                    num_features = 1
                
                if feature_set == "installments":
                    num_features = 14

                X_part = np.empty((1, num_features))
                X_part.fill(np.nan)
            else:
                transactions_df = part_transactions_df.loc[ix]
                
                if feature_set == "authorized_flag":
                    authorized_flag = transactions_df.authorized_flag
                    X_part = np.array([[np.mean(authorized_flag.values)]])

                if feature_set == "purchase_amount":
                    agg = dict(purchase_amount=("min", "mean", "max"))
                    X_part = transactions_df.agg(agg).T.values
                
                if feature_set == "purchase_amount_by_authorized_flag":
                    agg = dict(purchase_amount=("min", "mean", "max"))
                    x1 = transactions_df[transactions_df.authorized_flag == 1].agg(agg).T.values
                    x2 = transactions_df[transactions_df.authorized_flag == 0].agg(agg).T.values
                    X_part = np.concatenate((x1, x2), axis=1)
                
                if feature_set == "transactions_count":
                    X_part = np.array([[transactions_df.shape[0]]])
                
                if feature_set == "installments":
                    installments = transactions_df[["installments"]]
                    installments = pd.get_dummies(installments).values
                    X_part = np.mean(installments, axis=0).reshape(-1, 14)
                    
            X_parts.append(X_part)
        
        X.append(np.concatenate(X_parts))

        if clazz == "train":
            y.append(customer.target)
            
        max_num_customers -= 1
        
    return X, y

def process_train_part(part): return process_part(part, "train")
def process_test_part(part):  return process_part(part, "test")

# X, y = process_train_part(13);

# +
# %%time

from multiprocessing import Pool

with Pool(8) as pool: results = pool.map(process_train_part, range(TRANSACTIONS_N_PARTS))

# +
X = list(x[0] for x in results)
X = np.concatenate(X)

y = list(x[1] for x in results)
y = np.concatenate(y)

# +
TARGET_DIR_NAME = "development" if development else "production"

np.save(f"../data/3-feature-engineered/{TARGET_DIR_NAME}/train/{feature_set}.npy", X)

# np.save(f"../data/3-feature-engineered/{TARGET_DIR_NAME}/train/y.npy", y)
