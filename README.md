# Multi-class Serovar prediction of Salmonella strains.

## Overview
This repository contains a machine learning project focused on multi-class classification of serovars for Salmonella strains based on their nutrient utilization profiles. Its based on the data from the following paper: Seif et al, 2018. https://doi.org/10.1038/s41467-018-06112-5.

This project is designed to be used in conjuction with data generated from Genome-scale Metabolic Models of Salmonella strains. Specifically, nutrient utilization data for carbon, nitrogen, phosphorous and sulfur sources, under aerobic and anaerobic conditions ([see feature description](#about-the-data-and-features)).

The project contains two main components:

1. **Training Pipeline**: Automates the process of model training, including cross-validation, hyperparameter tuning, and model evaluation.
2. **Model Serving**: Facilitates model deployment through a Flask-based API, containerized using Docker.

The system is designed to be highly configurable, adaptable to various model architectures, and easy to deploy for practical use.

## Getting Started

### Contents

1. [Prerequisites](#Prerequisistes)
2. [Installation](#Installation)
3. [Model serving](#Model-serving)
4. [Training pipeline](#Training-pipeline)
5. [About the data and features](#about-the-data-and-features)

### Prerequisistes
The project was built using the following dependencies and versions:

* Python 3.11
* Poetry 1.7.1
* Docker (25.0.0) & Docker-desktop (4.17.0) 

### Installation
1. Clone the repository and change directory:
```
git clone https://github.com/CarlosFocil/SerovarClassification.git

cd SerovarClassification
```

2. Install required dependencies:
```
poetry install --no-root
```

3. Activate the environment
```
poetry shell
```

### Model serving

1. Build the docker image
Inside the main directory run the following command:
```
docker build -t serovar_prediction_service .
```
2. Run the container
```
docker run --rm -p 9696:9696 serovar_prediction_service
```
When succesfully running you should see the following message:
```
Model LabelEncoder_Pipeline_random_forest_SerovarClassifier_v1.bin loaded successfully. Ready to recieve requests and make predictions.
```
3. Test the service
```
cd inference/

python test_prediction_request.py
```

### Training models
The project contains an automated pipeline for training different models (Currently only random forest and decision trees supported).
All parameters for running the pipeline can be configured by changing the config.py file

1. Change to the training directory
```
cd training
```
2. Specify the paramaters for the pipeline in 'config.py' ([see below for description](#configuration-file-parameters))

3. Run the training script:
```
python train.py
```
This will automatically handle cross-validation, hyperparameter tuning and will select the best parameters. The model pipeline, together with the label encoder will be saved at the 'models' directory as a binary file.

### Configuration File Parameters

The configuration file is divided into three main sections: `PATH_SETTINGS`, `MODEL_SETTINGS`, and `PARAM_GRID`. Each section contains key-value pairs that are essential for the setup and execution of the model. Understanding these parameters is crucial for effective use of the software.

#### PATH_SETTINGS
This section defines the file paths used in the model training process.

- `training_data_path`: Specifies the relative path to the training dataset. The default path is `../data/prepared_data/complete_dataset.csv`, which points to a CSV file containing the prepared data.
- `output_model_path`: Indicates the relative path where the trained model will be saved. By default, it points to `../models`.

#### MODEL_SETTINGS
This section contains settings related to the model's configuration.

- `model_type`: Defines the type of model to be used. Currently set to `random_forest`, indicating that the model is a Random Forest classifier.
- `random_state`: A seed value for random number generation to ensure reproducibility. Set to `11`.
- `cv_folds`: Specifies the number of cross-validation folds to use. Set to `5`, which means the data will be split into 5 parts for cross-validation purposes.
- `model_version`: A string to indicate the version of the model, set to `v1`.
- `save_model`: A boolean indicating whether to save the trained model. Set to `False` by default.

#### PARAM_GRID
This section is used for hyperparameter tuning and defines a grid of parameters to be explored during model training.

- For the `random_forest` model, the following hyperparameters are configurable:
  - `feature_selection__threshold`: A list of variance thresholds for feature selection. Includes values `[0.1, 0.03, 0.01, 0.001]`.
  - `classifier__n_estimators`: Number of trees in the forest. Options include `[10, 50, 100]`.
  - `classifier__max_depth`: The maximum depth of the trees. Options are `[None, 1, 2, 3, 4]`, where `None` means no limit.
  - `classifier__min_samples_leaf`: The minimum number of samples required to be at a leaf node. Options are `[1, 2, 3]`.
  - `classifier__class_weight`: Weights associated with classes. Can be `balanced` or `'balanced_subsample'`.

- For the `decision_tree` model, the following hyperparameters are configurable:
  - `feature_selection__threshold`: A list of variance thresholds for feature selection. Includes values `[0.1, 0.03, 0.01, 0.001]`.
  - `classifier__criterion`: Criterion used for splitting. Options are `['gini', 'entropy', 'log_loss']`.
  - `classifier__max_depth`: The maximum depth of the tree. Options are `[None, 1, 2, 3, 4, 6, 8]`, where `None` indicates no maximum depth.
  - `classifier__min_samples_leaf`: The minimum number of samples required to be at a leaf node. Options are `[1, 2, 3, 4]`.

## About the data and features.

The data used in this project is located at 'data/prepared_data/six_classes_nutrient_profile_data.csv' which is a preprocessed dataset from the original article. The raw data is also included at 'data/raw_data/'.

The metabolic fluxes of different nutrients are used as features for training the models. There are 6 types of features in the data which are encoded as prefixes and suffixes on the feature names:

Prefix: Type of source.

`c_`: Carbon source

`n_`: Nitrogen source

`p_`: Phosphorus source

`s_`: Sulfur source

Suffix: Growth condition.

`(O2+)`: Fluxes for the given source on aerobic conditions.

`(O2-)`: Fluxes for the given source on anaerobic conditions.