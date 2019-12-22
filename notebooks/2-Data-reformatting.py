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

SOURCE_PATH = f'{DATA_PATH}/1-splitted'
TARGET_PATH = f'{DATA_PATH}/2-reformatted'

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/authorized_flag.csv', false_values=['N'], true_values=['Y'])

df.dtypes

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/authorized_flag.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/authorized_flag.csv', false_values=['N'], true_values=['Y'])

df.dtypes

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/authorized_flag.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/card_id.csv')

df.info()

# %time df.card_id = df.card_id.astype('category')

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/card_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/card_id.csv')

df.info()

# %time df.card_id = df.card_id.astype('category')

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/card_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/category_1.csv', false_values=['N'], true_values=['Y'])

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/category_1.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/category_1.csv', false_values=['N'], true_values=['Y'])

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/category_1.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/category_2.csv')

df.category_2.value_counts()

df.info()

# %time df['category_2_null'] = pd.isnull(df.category_2)

df.category_2_null.mean()

# %time df.category_2 = df.category_2.fillna(1).astype('int8')

df.category_2.value_counts()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/category_2.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/category_2.csv')

df.category_2.value_counts()

df.info()

# %time df['category_2_null'] = pd.isnull(df.category_2)

df.category_2_null.mean()

# %time df.category_2 = df.category_2.fillna(1).astype('int8')

df.category_2.value_counts()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/category_2.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/category_3.csv')

df.category_3.value_counts()

df.info()

# %time df['category_3_null'] = pd.isnull(df.category_3)

df.category_3_null.mean()

# %time df.category_3 = df.category_3.fillna('A').map({'A': 1, 'B': 2, 'C': 3}).astype('int8')

df.category_3.value_counts()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/category_3.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/category_3.csv')

df.category_3.value_counts()

df.info()

# %time df['category_3_null'] = pd.isnull(df.category_3)

df.category_3_null.mean()

# %time df.category_3 = df.category_3.fillna('A').map({'A': 1, 'B': 2, 'C': 3}).astype('int8')

df.category_3.value_counts()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/category_3.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/city_id.csv')

df.city_id.min(), df.city_id.max()

df.info()

# %time df.city_id = df.city_id.astype('int16')

df.city_id.min(), df.city_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/city_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/city_id.csv')

df.city_id.min(), df.city_id.max()

df.info()

# %time df.city_id = df.city_id.astype('int16')

df.city_id.min(), df.city_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/city_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/installments.csv')

df.installments.min(), df.installments.max()

df.info()

# %time df.installments = df.installments.astype('int16')

df.installments.min(), df.installments.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/installments.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/installments.csv')

df.installments.min(), df.installments.max()

df.info()

# %time df.installments = df.installments.astype('int16')

df.installments.min(), df.installments.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/installments.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/merchant_category_id.csv')

df.merchant_category_id.min(), df.merchant_category_id.max()

df.info()

# %time df.merchant_category_id = df.merchant_category_id.astype('int16')

df.merchant_category_id.min(), df.merchant_category_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/merchant_category_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/merchant_category_id.csv')

df.merchant_category_id.min(), df.merchant_category_id.max()

df.info()

# %time df.merchant_category_id = df.merchant_category_id.astype('int16')

df.merchant_category_id.min(), df.merchant_category_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/merchant_category_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/merchant_id.csv')

df.info()

# %time df.merchant_id.value_counts().head()

# %time df['merchant_id_null'] = pd.isnull(df.merchant_id)

df.merchant_id_null.mean()

# %time df.merchant_id = df.merchant_id.fillna('M_ID_00a6ca8a8a').astype('category')

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/merchant_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/merchant_id.csv')

df.info()

# %time df.merchant_id.value_counts().head()

# %time df['merchant_id_null'] = pd.isnull(df.merchant_id)

df.merchant_id_null.mean()

# %time df.merchant_id = df.merchant_id.fillna('M_ID_00a6ca8a8a').astype('category')

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/merchant_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/month_lag.csv')

df.month_lag.min(), df.month_lag.max()

df.info()

# %time df.month_lag = df.month_lag.astype('int8')

df.month_lag.min(), df.month_lag.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/month_lag.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/month_lag.csv')

df.month_lag.min(), df.month_lag.max()

df.info()

# %time df.month_lag = df.month_lag.astype('int8')

df.month_lag.min(), df.month_lag.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/month_lag.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/purchase_amount.csv')

df.purchase_amount.min(), df.purchase_amount.max()

df.info()

# %time df.purchase_amount = df.purchase_amount.astype('float32')

df.purchase_amount.min(), df.purchase_amount.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/purchase_amount.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/purchase_amount.csv')

df.purchase_amount.min(), df.purchase_amount.max()

df.info()

# %time df.purchase_amount = df.purchase_amount.astype('float32')

df.purchase_amount.min(), df.purchase_amount.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/purchase_amount.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/purchase_date.csv', parse_dates=['purchase_date'])

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/purchase_date.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/purchase_date.csv', parse_dates=['purchase_date'])

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/purchase_date.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/state_id.csv')

df.state_id.min(), df.state_id.max()

df.info()

# %time df.state_id = df.state_id.astype('int8')

df.state_id.min(), df.state_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/state_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/state_id.csv')

df.state_id.min(), df.state_id.max()

df.info()

# %time df.state_id = df.state_id.astype('int8')

df.state_id.min(), df.state_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/state_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/historical_transactions/subsector_id.csv')

df.subsector_id.min(), df.subsector_id.max()

df.info()

# %time df.subsector_id = df.subsector_id.astype('int8')

df.subsector_id.min(), df.subsector_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/historical_transactions/subsector_id.feather')

# ***

# %time df = pd.read_csv(f'{SOURCE_PATH}/new_merchant_transactions/subsector_id.csv')

df.subsector_id.min(), df.subsector_id.max()

df.info()

# %time df.subsector_id = df.subsector_id.astype('int8')

df.subsector_id.min(), df.subsector_id.max()

df.info()

df.isnull().sum()

df.to_feather(f'{TARGET_PATH}/new_merchant_transactions/subsector_id.feather')
