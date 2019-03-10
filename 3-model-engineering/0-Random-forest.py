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
    feature_matrix_df = feature_matrix_df.replace([np.inf, -np.inf], np.nan)
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
    for col in cols_with_nulls: feature_matrix_df[col].fillna(feature_matrix_df[col].mode()[0], inplace=True)
    print(f"{feature_matrix_clazz.upper()} has {feature_matrix_df.isnull().sum().sum()} missing values!")

# +
X_train, y_train = feature_matrix_dfs["train"].drop("target", axis=1), feature_matrix_dfs["train"].target

X_test = feature_matrix_dfs["test"].drop("target", axis=1)

# +
# %%time

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_features="log2", max_depth=13, n_jobs=-1, random_state=13)
scores = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=3))
print("CV: %.4f +/- %.4f" % (scores.mean(), scores.std() ** 2))

# +
# %%time

model.fit(X_train, y_train)
# -

plt.scatter(y_train, model.predict(X_train), s=0.25);

# +
# %%time

y_test = model.predict(X_test)

# +
submission_df = feature_matrix_dfs["test"].assign(target=y_test).loc[:, ["target"]]

display(submission_df)
    
submission_df.to_csv("../submission.csv")
