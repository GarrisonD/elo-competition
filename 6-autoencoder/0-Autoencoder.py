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

# +
import random
import pandas as pd
pd.options.display.max_columns = 0

import matplotlib.pyplot as plt

# %matplotlib inline

# %config InlineBackend.figure_format = "retina"

# +
import os

parts_dir = "../data/1-feature-engineering"

parts_names = os.listdir(parts_dir)
parts_names = filter(lambda x: not x.startswith("."), parts_names)
parts_paths = sorted(map(lambda part_name: parts_dir + "/" + part_name, parts_names))

parts_paths[:5]

# +
# %%time

parts_dfs = tuple(map(pd.read_csv, parts_paths))

with pd.option_context("display.max_rows", 6): display(random.choice(parts_dfs))

# +
# %%time

feature_matrix = pd.concat(parts_dfs, ignore_index=True, sort=False)

with pd.option_context("display.max_rows", 6): display(feature_matrix)

# +
customers = feature_matrix.drop("MODE(transactions.merchant_id)", axis=1).fillna(0)

customers_ord = customers[customers.target > -33].sample(2000)
customers_ano = customers[customers.target < -33].sample(2000)

# +
# %%time

from sklearn.manifold import TSNE

X = pd.concat((customers_ano, customers_ord))
y = (X.target < -33).astype(int)
X = X.drop("target", axis=1)

X_t = TSNE(random_state=13).fit_transform(X)

# +
X_t_ord = X_t[y == 0]
X_t_ano = X_t[y == 1]

plt.scatter(X_t_ord[:, 0], X_t_ord[:, 1], marker='o', color='g', linewidth='0.5', alpha=0.8, label='Ordinary')
plt.scatter(X_t_ano[:, 0], X_t_ano[:, 1], marker='o', color='r', linewidth='0.5', alpha=0.8, label='Anomaly')

plt.legend(loc='best')
plt.show()

# +
from keras.layers import Input, Dense
import keras.regularizers as regularizers

## input layer 
input_layer = Input(shape=(X.shape[1],))

## encoding part
encoded = Dense(256, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(128, activation='relu')(encoded)

## decoding part
decoded = Dense(128, activation='tanh')(encoded)
decoded = Dense(256, activation='tanh')(decoded)

## output layer
output_layer = Dense(X.shape[1], activation='relu')(decoded)

# +
from keras.models import Model

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")

# +
from sklearn.preprocessing import MinMaxScaler

X_scaled = MinMaxScaler().fit_transform(X.values)
X_scaled_ord = X_scaled[y == 0]
X_scaled_ano = X_scaled[y == 1]
# -

autoencoder.fit(X_scaled_ord, X_scaled_ord, 
                batch_size=64, epochs=35,
                shuffle=True,
                validation_split=0.20);

# +
from keras.models import Sequential

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
# -

ord_hid_rep = hidden_representation.predict(X_scaled_ord)
ano_hid_rep = hidden_representation.predict(X_scaled_ano)

# +
import numpy as np

rep_x = np.append(ord_hid_rep, ano_hid_rep, axis = 0)

# +
# %%time

X_t = TSNE(random_state=13).fit_transform(rep_x)

# +
X_t_ord = X_t[y == 0]
X_t_ano = X_t[y == 1]

plt.scatter(X_t_ord[:, 0], X_t_ord[:, 1], marker='o', color='g', linewidth='0.5', alpha=0.8, label='Ordinary')
plt.scatter(X_t_ano[:, 0], X_t_ano[:, 1], marker='o', color='r', linewidth='0.5', alpha=0.8, label='Anomaly')

plt.legend(loc='best')
plt.show()
