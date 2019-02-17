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
X_train, y_train = feature_matrix_dfs["train"].drop("target", axis=1), feature_matrix_dfs["train"].target

X_train_ordinal, y_train_ordinal = X_train[y_train > -33], y_train[y_train > -33]
X_train_anomaly, y_train_anomaly = X_train[y_train < -33], y_train[y_train < -33]

X_test = feature_matrix_dfs["test"].drop("target", axis=1)

# +
# %%time

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_features="log2", max_depth=10, n_jobs=-1, random_state=13)
scores = np.sqrt(-cross_val_score(model, X_train_ordinal, y_train_ordinal, scoring="neg_mean_squared_error"))
print("CV: %.4f +/- %.4f" % (scores.mean(), scores.std() ** 2))

# +
# %%time

model.fit(X_train_ordinal, y_train_ordinal)
# -

plt.scatter(y_train, model.predict(X_train), s=0.25);

# +
# %%time

y_test = model.predict(X_test)

# +
# %%time

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

anomaly_classifier = RandomForestClassifier(n_estimators=200,
                                            max_features="log2",
                                            random_state=13,
                                            max_depth=10,
                                            n_jobs=-1)

X_train, y_train = shuffle(X_train, y_train)
y_train = (y_train < -33).astype(np.int)

scores = cross_val_score(anomaly_classifier, X_train, y_train, cv=5)
print("CV: %.4f +/- %.4f" % (scores.mean(), scores.std() ** 2))

# +
# %%time

anomaly_classifier.fit(X_train, y_train)
mask = anomaly_classifier.predict(X_test)

y_test[mask == 1] = feature_matrix_dfs["train"].target.min()

# +
submission_df = feature_matrix_dfs["test"].assign(target=y_test).loc[:, ["target"]]

display(submission_df)
    
submission_df.to_csv("../submission.csv")
