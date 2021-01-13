Information Gain with Registration
==============================

Information gain toolbox for multimodal dataset processing

### Metacentrum RUN

0) enable ssh agent
```bash
ssh-add
```
1) ssh to metacentrum
```bash
sshm alfrid
```
2) clone repository
```bash
git clone git@github.com:gimlidc/igre.git
```
3) install aws cli
```bash
module add python-3.6.2-gcc
pip3 install --user awscli
# Add path to user bin into your .bash_profile
```
4) configure aws profile
```bash
aws configure --profile igre
# follow requested ...
```
5) sync data from s3
```bash
make sync_data_from_s3
```
6) run your scripts
- customize your clusterize-configuration.yml
- setup directory for outputs into igre/data/processed folder
- install dependencies
```bash
clusterize submit
```
7) sync outputs to s3
```bash
make sync_data_to_s3
```


### IGRE test
```bash
git clone git@github.com:gimlidc/igre
cd igre
make sync_data_from_s3
pipenv run python src/models/igre-test.py
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
