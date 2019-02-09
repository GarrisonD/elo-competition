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

# %run ../0-utils/0-Base.ipynb

# +
PARTS_DIR_PATH = "../data/2-formatted"

def read_part_customers_df(part, clazz):
    return pd.read_csv(f"{PARTS_DIR_PATH}/%03d/{clazz}.csv" % part)

def read_part_transactions_df(part):
    return pd.read_csv(f"{PARTS_DIR_PATH}/%03d/transactions.csv" % part, parse_dates=["purchase_date"])
# -

dates = pd.date_range(start="2017-01", end="2018-04", freq="M").to_period("M"); display(dates)

agg = dict(purchase_amount=("min", "mean", "max"))

# +
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
        if clazz == "train" and max_num_customers < 0 and customer.target > -33: continue
        
        X_parts = []
        
        for date in dates:
            ix = (customer.card_id, date.year, date.month)
            
            if not part_transactions_df.index.contains(ix):
                X_part = np.empty((1, 9))
                X_part.fill(np.nan)
            else:
                transactions_df = part_transactions_df.loc[ix]
                
                x1 = transactions_df.agg(agg).T.values
                x2 = transactions_df[transactions_df.authorized_flag == "Y"].agg(agg).T.values
                x3 = transactions_df[transactions_df.authorized_flag == "N"].agg(agg).T.values
                
                X_part = np.concatenate((x1, x2, x3), axis=1)
                
            X_parts.append(X_part)
        
        X.append(np.concatenate(X_parts))

        if clazz == "train":
            y.append(customer.target)
            
        max_num_customers -= 1
        
    return X, y

def process_train_part(part): return process_part(part, "train")
def process_test_part(part):  return process_part(part, "test")

# process_train_part(11);

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
from sklearn.preprocessing import MinMaxScaler

X_origin_shape = X.shape

X = MinMaxScaler().fit_transform(X.reshape(-1, 9))

# for months without any data
X[np.isnan(X).all(axis=1)] = -1

# for months without partial data
X[np.isnan(X)] = 0

X = X.reshape(*X_origin_shape)
# -

np.save("../data/2-feature-engineered/X_train.npy", X)
np.save("../data/2-feature-engineered/y_train.npy", y)
