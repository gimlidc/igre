Information Gain with Registration
==============================

Information gain toolbox for multimodal dataset processing. There are several starting points of this project:

* In [models](./src/models) where ANN [construction](./src/models/igre.py), [testing](./src/models/igre_test.py) as well as [benchmarking](./src/models/igre_test_batch.py) are present.
* In [notebooks](./notebooks) folder are mainly evaluations of benchmarks, but also other stuff.

## IGRE architecture 

TODO: Please keep this in sync with current git branch!!!
Current IGRE architecture contains layers for:
* shift estimation
* rotation estimation
* scale estimation
<!-- * modality transformation -->

## IGRE test architecture
Our test consist of:

1. data preparation
   - load mat file
   - crop required section
   - select input and output modalities
   - transform input modality according to configuration (shift, affine, ...)
2. IGRE run (see above)
3. Printout into output file(s) 

NOTE: Printout structure must match expected structure in aggregation function [./src/data/raw/process_raw_results.py], please change it in sync. 

## IGRE batch test

Main purpose is to run IGRE with different parametrization. This means various ANN layer configuration as well as various inputs. Multiple runs are performed thru Metacentrum as well as parallelization.

#### L1: More than one machine
This L1 parallelization is performed by ''clusterize'' tool, which takes configuration yaml (e.g. at [./data/interim/examples/clusterize-configuration.yml]) and creates parametrized Metacentrum jobs.

Typical usage is for configuration of input image transformation.

#### L2: Several runs at one machine
Because our first tests shown that IGRE for one image with one parametrization is too short (for being Metacentrum job) we adapt igre-test for a batch run (@see [./src/models/igre_test_batch.py]). On this level yaml configuration for igre-test is fulfilled by batch parameters ... e.g.:
```yaml
batch:
  params: ["output_dimension", "matfile"] # list of batch parameters (will be overwritten in conf above)
  output_dimension: # name of overwritten parameter, these are processed in igre_test_batch.py
    min: 10 # starting value
    max: 23 # end value
    step: 2 # step value
  matfile: # again overwritten parameter, here is replaced value constructed from template and regex (in igre_test_batch)
    template: "sample_{id}.mat"
    replace: "{id}"
    min: 1
    max: 44
``` 
According to this configuration multiple config dicts are generated (and possibly run in parallel).

Both levels should fit to:
1. Metacentrum maximum jobs in queue (cca 7k)
1. Time limit for a job (the queue must be selected accordingly: 24h, 48h, 96h)

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
<!--
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
-->
3) run your scripts
- install [clusterize](https://github.com/jakubtucek/clusterize)
- customize your clusterize-configuration.yml (in current pwd or specify path)
- setup directory for outputs (e.g. into igre/data/processed folder)
- install dependencies
```bash
clusterize submit
```
4) sync aggregated outputs to s3, used configurations, etc.
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
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
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
