from ml_utils import grid_search_variance_selection, load_and_split_data
from config import *

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import logging
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Read config settings
    logging.info('Preparing config settings...')
    paths = PATH_SETTINGS
    model_settings = MODEL_SETTINGS
    hyperparam_grid = PARAM_GRID

    # Read file and prepare training data
    logging.info('Loading data...')
    df_train, df_test, y_train, y_test = load_and_split_data(paths["training_data_path"],
                                                             size=0.2, 
                                                             target_variable='serotype', 
                                                             rand_state=model_settings['random_state']
                                                             )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Setting up model
    model_type = model_settings["model_type"]
    cv_folds = model_settings["cv_folds"]

    if model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=model_settings["random_state"], n_jobs=-1)
        param_grid = hyperparam_grid[model_type]
    
    # Hyperparameter tuning and cross validation
    logging.info(f"Performing hyperparameter tuning and cross-validation with the following settings...")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Hyperparameters {param_grid}, cross-validation folds: {cv_folds}")

    grid_search = grid_search_variance_selection(df_train, y_train_encoded,
                                                           classifier, param_grid,
                                                           select_threshold=0.01,
                                                           cv_folds=cv_folds
                                                           )

    logging.info(f"Best parameters found {grid_search.best_params_}")
    logging.info(f"Best accuracy: {grid_search.best_score_}")
    
    
    # Testing final model based on best hyperparameters   
    logging.info('Testing final model...') 
    y_pred_test_encoded = grid_search.predict(df_test)
    y_pred_test = label_encoder.inverse_transform(y_pred_test_encoded)
    logging.info(f"Accuracy on test data: {accuracy_score(y_test, y_pred_test)}")

    # Save model
    if model_settings['save_model']:
        
        out_path = paths['output_model_path']
        model_version = model_settings['model_version']
        best_model_pipeline = grid_search.best_estimator_

        output_file = os.path.join(out_path, f"LabelEncoder_Pipeline_{model_type}_SerovarClassifier_{model_version}.bin")

        with open(output_file,'wb') as f_out:
            pickle.dump((label_encoder,best_model_pipeline),f_out)
        logging.info(f'Model is saved as {output_file}')

if __name__=='__main__':
    main()