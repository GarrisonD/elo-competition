#### Getting started:

1. Install `Anaconda` / `Miniconda`

2. Run `conda env create` to install all the required `Python` packages

4. Run `conda activate elo-competition` to select the created environment

3. For `tqdm` to work properly run:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

4. For `pandas` to work properly run:

```bash
pip install pyarrow
```

5. For `tensorboard` to work properly run:

```bash
pip install future tensorboard==1.15.0
```

6. Run `python setup.py develop` to install the repository as a `Python` package. [Read more](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/)

7. Fetch the data from Kaggle:

```bash
kaggle competitions download -c elo-merchant-category-recommendation -p data/raw

unzip data/raw/elo-merchant-category-recommendation.zip -d data/raw

rm data/raw/elo-merchant-category-recommendation.zip
```

8. Run `jupyter lab --notebook-dir notebooks`

9. Run all the notebooks in the order they present to reproduce the results

10. (*Optional*) To run tests run `python setup.py test`
