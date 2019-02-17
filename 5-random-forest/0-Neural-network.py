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

# Read data that will be used for training / testing:

# +
# %%time

feature_matrix_dfs = dict()

for feature_matrix_clazz in ("train", "test"):
    feature_matrix_df = pd.read_csv(f"../data/2-feature-engineered/{feature_matrix_clazz}.csv")
    feature_matrix_df = feature_matrix_df.drop("MODE(transactions.merchant_id)", axis=1)
    feature_matrix_df = feature_matrix_df.set_index("card_id")
    
    display(feature_matrix_df)

    feature_matrix_dfs[feature_matrix_clazz] = feature_matrix_df
# -

# Check for missing values:

for feature_matrix_clazz, feature_matrix_df in feature_matrix_dfs.items():
    print(feature_matrix_clazz.upper() + ":")
    
    cols_with_nulls = feature_matrix_df.columns[feature_matrix_df.isnull().any()].tolist()
    
    display(cols_with_nulls)

# Get rid of missing values:

for feature_matrix_clazz, feature_matrix_df in feature_matrix_dfs.items():
    cols_with_nulls = feature_matrix_df.columns[feature_matrix_df.isnull().any()].tolist()
    
    for skew_col in filter(lambda x: x.startswith("SKEW"), cols_with_nulls):
        feature_matrix_df[skew_col].fillna(feature_matrix_df[skew_col].mode()[0], inplace=True)
    
    print(f"{feature_matrix_clazz.upper()} has {feature_matrix_df.isnull().sum().sum()} missing values!")

# +
from sklearn.preprocessing import StandardScaler

X_train, y_train = feature_matrix_dfs["train"].drop("target", axis=1), feature_matrix_dfs["train"].target

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = feature_matrix_dfs["test"].drop("target", axis=1)
X_test = scaler.transform(X_test)

# +
# %%time

from keras.models import Sequential
from keras.layers import Dense, Dropout

regressor = Sequential()

regressor.add(Dense(512, input_dim=X_train.shape[1], activation="relu"))
regressor.add(Dropout(0.5))

regressor.add(Dense(1024, activation="relu"))
regressor.add(Dropout(0.5))

regressor.add(Dense(512, activation="relu"))
regressor.add(Dropout(0.5))

regressor.add(Dense(256, activation="relu"))
regressor.add(Dropout(0.5))

regressor.add(Dense(1))

regressor.compile(optimizer="adam", loss="mse")

regressor.fit(X_train, y_train, batch_size=256, epochs=5)
# -

plt.scatter(y_train, regressor.predict(X_train), s=0.25);

# +
# %%time

y_test = regressor.predict(X_test)

# +
submission_df = feature_matrix_dfs["test"].assign(target=y_test).loc[:, ["target"]]

display(submission_df)
    
submission_df.to_csv("../submission.csv")
