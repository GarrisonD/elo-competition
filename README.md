```bash
kaggle competitions download -c elo-merchant-category-recommendation -p data/raw

unzip elo-merchant-category-recommendation.zip && rm elo-merchant-category-recommendation.zip

chmod 600 data/raw/*
```

For `tqdm`:

```bash
jupyter nbextension enable --py widgetsnbextension

jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

For `tensorboard`:

```bash
pip install future tensorboard==1.15.0
```
