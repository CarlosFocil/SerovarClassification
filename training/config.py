PATH_SETTINGS = {
    'training_data_path' : '../data/prepared_data/complete_dataset.csv',
    'output_model_path' : '../models',
}

MODEL_SETTINGS = {
    'model_type':'random_forest',
    'random_state': 11,
    'cv_folds':5,
    'model_version':'v1',
    'save_model':False
}

PARAM_GRID = {
    'random_forest': {'feature_selection__threshold': [0.1, 0.03, 0.01, 0.001],
                      'classifier__n_estimators': [10,50,100],
                      'classifier__max_depth': [None, 1, 2 ,3 ,4],
                      'classifier__min_samples_leaf':[1,2,3],
                      'classifier__class_weight':['balanced', 'balanced_subsample']
                     },
    'decision_tree': {'feature_selection__threshold': [0.1, 0.03, 0.01, 0.001],
                      'classifier__criterion': ['gini','entropy','log_loss'],
                      'classifier__max_depth': [None, 1, 2, 3, 4, 6 ,8],
                      'classifier__min_samples_leaf': [1,2,3,4]
                      }
}