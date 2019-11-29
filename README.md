In order to fetch the data from Kaggle:

```bash
kaggle competitions download -c elo-merchant-category-recommendation -p data/raw

unzip data/raw/elo-merchant-category-recommendation.zip -d data/raw

rm data/raw/elo-merchant-category-recommendation.zip

chmod 600 data/raw/*
```

For `tqdm` to work properly:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

For `pandas` to work properly:

```bash
pip install pyarrow
```

For `tensorboard` to work properly:

```bash
pip install future tensorboard==1.15.0
```
