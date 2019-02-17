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
# %%time

feature_matrix_df = pd.read_csv(f"../data/2-feature-engineered/train.csv")

ordinal_df = feature_matrix_df[feature_matrix_df.target > -30]
anomaly_df = feature_matrix_df[feature_matrix_df.target < -30]

with pd.option_context("display.max_rows", 6): display(ordinal_df, anomaly_df)

# +
import re

installments_cols = list(filter(lambda x: "PERCENT_TRUE" in x and "installments_" in x, feature_matrix_df.columns))

n_groups = len(installments_cols)

means_ordinal = ordinal_df[installments_cols].mean(axis=0).values
stds_ordinal  = ordinal_df[installments_cols].std(axis=0).values

means_anomaly = anomaly_df[installments_cols].mean(axis=0).values
stds_anomaly  = anomaly_df[installments_cols].std(axis=0).values

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {"ecolor": "0.3"}

rects1 = ax.bar(index, means_ordinal, bar_width,
                alpha=opacity, color="b",
                yerr=stds_ordinal, error_kw=error_config,
                label="Ordinal")

rects2 = ax.bar(index + bar_width, means_anomaly, bar_width,
                alpha=opacity, color="r",
                yerr=stds_anomaly, error_kw=error_config,
                label="Anomaly")

ax.set_title("Installments")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(list(map(lambda x: re.search("(\-?\d+)", x).group(1), installments_cols)))
ax.legend()

plt.show()
