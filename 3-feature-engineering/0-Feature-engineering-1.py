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

# Read data about customers (`train` and `test`):

# +
customers_dfs = dict()

for clazz in ("train", "test"):
    df = pd.read_csv(f"../data/raw/{clazz}.csv", parse_dates=["first_active_month"])

    # test.csv has one missing value - replace it with the most popular 
    # first_active_month for rows in train.csv that have the same feature_x
    if "test" == clazz: df.fillna(pd.to_datetime("2017-01"), inplace=True)
        
    if "target" not in df.columns: df["target"] = 0

    feature_1_backup = df.feature_1
    feature_2_backup = df.feature_2

    df = pd.get_dummies(df, columns=["feature_1", "feature_2"])

    df["feature_1"] = feature_1_backup
    df["feature_2"] = feature_2_backup
    
    display(df)
    
    customers_dfs[clazz] = df
# -

# Define a sample partition of customers:

# +
sample_customers_df = add_part(customers_dfs["train"], n_parts=TRANSACTIONS_N_PARTS["new"])
sample_customers_df = sample_customers_df.loc[lambda x: x.part == 13].drop("part", axis=1)

display(sample_customers_df)
# -

# Define a function for reading and processing a partition with transactions:

# +
from datetime import datetime
from pandas.api.types import CategoricalDtype

transactions_cat_cols = ("city_id",
                         "category_3",
                         "merchant_category_id",
                         "state_id",
                         "subsector_id")

transactions_dtype = {cat_col: "category" for cat_col in transactions_cat_cols}

transactions_dtype["installments"] = CategoricalDtype(
    categories=np.arange(-1, 13).tolist() + [999],
    ordered=True,
)

def read_transactions_part(part_file_name):
    transactions_df = pd.read_csv(f"../data/1-partitioned/{part_file_name}",
                                  parse_dates=["purchase_date"],
                                  dtype=transactions_dtype)
    
    transactions_df = transactions_df.reset_index().rename(columns={"index": "ID"})

    days_diff = datetime.today() - transactions_df.purchase_date
    transactions_df["month_diff"] = days_diff // np.timedelta64(1, "M")
    transactions_df["month_diff"] += transactions_df.month_lag
    
    cat_cols_mapping = dict(
        authorized_flag=dict(
            Y=1,
            N=0,
        ),
        category_1=dict(
            Y=1,
            N=0,
        ),
        category_3=dict(
            A=0,
            B=1,
            C=2,
        ),
    )
    
    for col, mapping in cat_cols_mapping.items():
        transactions_df[col] = transactions_df[col].map(mapping)
    
    for col in ("authorized_flag", "category_1", "category_2", "category_3"):
        transactions_df[col] = transactions_df[col].fillna(-1).astype(int)
        
    category_2_backup = transactions_df.category_2
    category_3_backup = transactions_df.category_3
    installments_backup = transactions_df.installments

    transactions_df = pd.get_dummies(transactions_df, columns=["category_2", "category_3", "installments"])
    
    transactions_df["category_2"] = category_2_backup
    transactions_df["category_3"] = category_3_backup
    transactions_df["installments"] = installments_backup

    return transactions_df
# -

# Read a sample partition with `read_transactions_part(...)`:

# +
# %%time

sample_transactions_df = read_transactions_part("new.013.csv")

display(sample_transactions_df)

# +
# %%time

merchants_df = pd.read_csv("../data/raw/merchants.csv")
merchants_df = merchants_df.drop_duplicates("merchant_id")
merchants_df = merchants_df.reset_index().rename(columns={"index": "ID"})

cat_cols_mapping = dict(
    category_1=dict(
        Y=1,
        N=0,
    ),
    category_4=dict(
        Y=1,
        N=0,
    ),
    most_recent_sales_range=dict(
        A=0,
        B=1,
        C=2,
        D=3,
        E=4,
    ),
    most_recent_purchases_range=dict(
        A=0,
        B=1,
        C=2,
        D=3,
        E=4,
    ),
)

for col, mapping in cat_cols_mapping.items():
    merchants_df[col] = merchants_df[col].map(mapping)

for col in ("category_1", "category_2", "category_4", "most_recent_sales_range", "most_recent_purchases_range"):
    merchants_df[col] = merchants_df[col].fillna(-1).astype(int)

category_2_backup = merchants_df.category_2
most_recent_sales_range_backup = merchants_df.most_recent_sales_range
most_recent_purchases_range_backup = merchants_df.most_recent_purchases_range

merchants_df = pd.get_dummies(merchants_df, columns=["category_2",
                                                     "most_recent_sales_range",
                                                     "most_recent_purchases_range"])

merchants_df["category_2"] = category_2_backup
merchants_df["most_recent_sales_range"] = most_recent_sales_range_backup
merchants_df["most_recent_purchases_range"] = most_recent_purchases_range_backup

merchants_df = merchants_df.fillna(0)

display(merchants_df)
# -

# Define a function for building an entry set for `customers_df` and `transactions_df`:

# +
import featuretools as ft
import featuretools.variable_types as vtypes

transactions_vtypes = dict(authorized_flag=vtypes.Boolean,
                           category_1=vtypes.Boolean)

for ord_col in ("installments", "category_2", "category_3"):
    transactions_vtypes[ord_col] = vtypes.Ordinal
    
    for bool_col in filter(lambda x: x.startswith(ord_col + "_"), sample_transactions_df.columns):
        transactions_vtypes[bool_col] = vtypes.Boolean
        
customers_vtypes = {
    "feature_1": vtypes.Ordinal,
    "feature_2": vtypes.Ordinal,
    "feature_3": vtypes.Boolean,
}

merchants_vtypes = dict(category_1=vtypes.Boolean,
                        category_4=vtypes.Boolean)

for ord_col in ("category_2", "most_recent_sales_range", "most_recent_purchases_range"):
    merchants_vtypes[ord_col] = vtypes.Ordinal
    
    for bool_col in filter(lambda x: x.startswith(ord_col + "_"), merchants_df.columns):
        merchants_vtypes[bool_col] = vtypes.Boolean

def build_entity_set(customers_df, transactions_df, merchants_df):
    es = ft.EntitySet()
    
    es.entity_from_dataframe(entity_id="customers",
                             dataframe=customers_df,
                             index="card_id",
                             variable_types=customers_vtypes)

    es.entity_from_dataframe(entity_id="transactions",
                             dataframe=transactions_df,
                             index="ID",
                             variable_types=transactions_vtypes)
    
    es.entity_from_dataframe(entity_id="merchants",
                             dataframe=merchants_df,
                             index="merchant_id",
                             variable_types=merchants_vtypes)

    customer_transactions_relationship = ft.Relationship(es["customers"]["card_id"],
                                                         es["transactions"]["card_id"])
    
    merchants_transactions_relationship = ft.Relationship(es["merchants"]["merchant_id"],
                                                          es["transactions"]["merchant_id"])

    es.add_relationships([customer_transactions_relationship,
                          merchants_transactions_relationship])

    es["transactions"]["category_1"].interesting_values = [1]
    es["transactions"]["authorized_flag"].interesting_values = [1]

    for col in ("category_2", "category_3", "installments"):
        for dummy_col in filter(lambda x: x.startswith(col + "_"), transactions_df.columns):
            es["transactions"][dummy_col].interesting_values = [1]
            
    es["merchants"]["category_1"].interesting_values = [1]
    es["merchants"]["category_4"].interesting_values = [1]
    
    for col in ("category_2", "most_recent_sales_range", "most_recent_purchases_range"):
        for dummy_col in filter(lambda x: x.startswith(col + "_"), merchants_df.columns):
            es["merchants"][dummy_col].interesting_values = [1]

    return es
# -

# Build an entity set for sample data:

es = build_entity_set(sample_customers_df, sample_transactions_df, merchants_df)

# Do deep feature synthesis:

# +
features = ft.dfs(entityset=es, target_entity="customers", features_only=True)

print("Number of features: %d" % len(features))

display(features)

# +
feature_matrix_df = ft.calculate_feature_matrix(features=features, entityset=es, verbose=True)

display(feature_matrix_df)

# +
# %%time

from multiprocessing import Pool

def calculate_feature_matrix_for(transactions_part_file_name, customers_part_df):
    transactions_part_df = read_transactions_part(transactions_part_file_name)

    part_entity_set = build_entity_set(
        customers_part_df,
        transactions_part_df,
        merchants_df,
    )
    
    feature_matrix_part_df = ft.calculate_feature_matrix(
        features=features, entityset=part_entity_set,
    )
    
    return feature_matrix_part_df

feature_matrix_dfs = dict()

for customers_clazz, customers_df in customers_dfs.items():
    for transactions_clazz, n_parts in TRANSACTIONS_N_PARTS.items():
        customers_with_part_df = add_part(customers_df, n_parts=n_parts)
        
        processes_args = []
        
        for part, customers_part_df in customers_with_part_df.groupby("part"):
            transactions_part_file_name = "%s.%03d.csv" % (transactions_clazz, part)
            processes_args.append((transactions_part_file_name, customers_part_df))
        
        with Pool(8) as p:
            feature_matrix_part_dfs = p.starmap(
                calculate_feature_matrix_for,
                processes_args,
            )

        feature_matrix_df = pd.concat(
            feature_matrix_part_dfs,
            sort=False,
        )
        
        feature_matrix_dfs[customers_clazz] = feature_matrix_df
        
        break # only old transactions for now

# +
# %%time

for feature_matrix_clazz, feature_matrix_df in feature_matrix_dfs.items():
    feature_matrix_df.to_csv(f"../data/2-feature-engineered/{feature_matrix_clazz}.csv")

# +
context = ("display.max_rows",     0,
           "display.max_colwidth", 0,)

with pd.option_context(*context): display(ft.list_primitives())
# -

# Add a `fraud` boolean flag?

# +
authorized_card_rate = transactions_df.groupby("card_id").authorized_flag.mean().sort_values()

with pd.option_context("display.max_rows", 10): display(authorized_card_rate)
# -

# `999` installments looks like a `fraud` indicator too

transactions_df.groupby("installments").authorized_flag.mean()

# Use `np.log` for `purchase_amount`?

# +
_, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
axs[0].hist(transactions_df.purchase_amount)
axs[1].hist(np.log(transactions_df.purchase_amount + 1))

plt.show()
