Getting started
================

1. Install ``Anaconda`` / ``Miniconda``

2. Run ``conda env create`` to install all the required ``Python`` packages

3. Run ``conda activate elo-competition`` to select the created environment

4. For ``tqdm`` to work properly run:

    .. code:: bash

        jupyter labextension install @jupyter-widgets/jupyterlab-manager

5. For ``tensorboard`` to work properly run:

    .. code:: bash

        pip install future tensorboard==1.15.0

6. Run ``python setup.py develop`` to install the repository as a ``Python`` package. `Read more <https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/>`__

7. Fetch the data from Kaggle:

    .. code:: bash

        kaggle competitions download -c elo-merchant-category-recommendation -p data/raw

        unzip data/raw/elo-merchant-category-recommendation.zip -d data/raw

        rm data/raw/elo-merchant-category-recommendation.zip

8.  Run ``jupyter lab --notebook-dir notebooks``

9.  Run all the notebooks in the order they present to reproduce the results

10. *(Optional)* To run tests run ``python setup.py test``
