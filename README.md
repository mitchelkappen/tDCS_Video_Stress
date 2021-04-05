Stress Prediction through Facial Behaviour
==============================

This project is the implementation of a study into the prediction of physiological arousal through facial behaviour, as part of a research internship by Alexander Kern at the GHEP Lab of the University of Ghent.

Requirements
------------
In order to run the notebooks and scripts that are present in the notebook directory, one needs to have created a conda environment with the correct packages installed. One can run the following command in the anaconda prompt to create an enivornment:\
`conda create --name <myenv>`

After creating an anaconda environment environment one can run the following commands to install the required packages:\
`conda config --add channels conda-forge`\
`conda install numpy pandas neurokit2 pytables scikit-learn matplotlib tqdm`\
`pip install bioread`

The notebooks can be accessed using Jupyter lab, which can be installed in the conda environment as follows:\
`conda install jupyterlab`

Jupyter-lab can be run through the anaconda prompt with the following command:\
`jupyter-lab`

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
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Notebooks Directory Organization
----------------------
The naming convention in the notebooks directory is a number (execution ordering), the creator's initials and a short '-' delimited description, for example: '1-ak-physio-dataframe'.  The directory contains the following notebooks that are responsible for the different components of the study: 
* '1-ak-physio-dataframe': Establishing the DataFrames that hold the physiological signals.
* '2-ak-video-dataframe': Establishing the DataFrames that contain the video signal.
* '3-ak-target-variable': Computing information necessary for determining the target variable.
* '4-ak-window-sampling': Sampling the video and physiological DataFrames in windows.
* '5-ak-window-processing': Processing the sampled windows, adding age and gender information.
* '6-ak-modelling': Using the sampled windows as data samples for various models.

Besides these 6 notebooks, the folder also contains two scripts that contain function that are responsible for the computation of the features and targets. These are imported in '4-ak-window-sampling'.
* 'features': Computation of various features, concerning facial and eye movement.
* 'targetComputation': Computation of HRV and SCL target variables.

Finally, the script 'raw-video-extraction' is responsible for the extraction of facial information from the video recordings using OpenFace.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
