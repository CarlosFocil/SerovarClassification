import pandas as pd
import numpy as np

from typing import List, Dict, Any, Tuple

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


def load_and_split_data(training_data_path:str, size:float, target_variable:str, rand_state:int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load data from a CSV file and split it into training and test sets.

    Parameters
    ----------
    training_data_path : str
        The file path to the training data in CSV format.
    size : float
        The proportion of the dataset to include in the test split.
    target_variable : str
        The name of the target variable column in the dataset.
    rand_state : int
        The random state for reproducibility of the split.

    Returns
    -------
    df_train : pd.DataFrame
        The training subset of the dataset.
    df_test : pd.DataFrame
        The test subset of the dataset.
    y_train : pd.Series
        The target variable for the training set.
    y_test : pd.Series
        The target variable for the test set.
    """
    df = pd.read_csv(training_data_path, index_col=0)
    df_train, df_test = train_test_split(df, test_size=size, random_state=rand_state)

    y_train = df_train[target_variable].values    
    y_test = df_test[target_variable].values

    del df_train[target_variable]
    del df_test[target_variable]

    return df_train, df_test, y_train, y_test

def grid_search_variance_selection(
    X_train: pd.DataFrame, 
    y_train_encoded: np.ndarray, 
    classifier, 
    param_grid: Dict[str,Any],
    select_threshold:float,
    cv_folds: int = 5
    ):
    """
    Run a grid search with cross-validation.

    Parameters
    ----------
    X_train: pd.DataFrame
        Feature dataset.
    y_train_encoded: np.ndarray
        Target variable from training previously encoded (e.g. with LabelEncoder()).
    classifier: Estimator
        An instance of the classifier to use (e.g. RandomForestClassifier())
    param_grid: Dict[str, Any]
        The hyperparameter grid to search.
    cv_folds: int
        Number of folds for cross-validation.

    Returns
    -------
    A fitted GridSearchCV object.
    """
    pipeline = Pipeline([
        ('feature_selection', VarianceThreshold(threshold=select_threshold)),
        ('classifier', classifier)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train_encoded)
    
    return grid_search

def grid_search_stdev_selection(
    X: pd.DataFrame, 
    y: pd.Series, 
    numerical_features: List[str], 
    target_variable: str, 
    classifier, 
    param_grid: Dict[str, Any],
    select_threshold: float,
    cv_folds: int = 5
):
    """
    Run a grid search with cross-validation.

    Parameters
    ----------
    X: pd.DataFrame
        Feature dataset.
    y: pd.Series
        Target variable.
    numerical_features: List[str]
        List of numerical features to consider for selection.
    target_variable: str
        Name of the target variable.
    classifier: Estimator
        An instance of the classifier to use (e.g. RandomForestClassifier())
    param_grid: Dict[str, Any]
        The hyperparameter grid to search.
    cv_folds: int
        Number of folds for cross-validation.

    Returns
    -------
    A fitted GridSearchCV object.
    """
    pipeline = Pipeline([
        ('feature_selection', FeatureSelectorStdDev(numerical_features=numerical_features, 
                                                    selection_threshold=select_threshold, 
                                                    target_variable=target_variable)),
        ('classifier', classifier)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    return grid_search

class FeatureSelectorStdDev(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features: List[str], selection_threshold: float, target_variable: str) -> None:
        """
        Initialize the FeatureSelector.

        Args:
        -----
        numerical_features (List[str])
            List of names of numerical features to consider.

        selection_threshold (float): 
            The percentile threshold for feature selection based on standard deviation.

        target_variable (str): 
            Name of the target variable.

        Attributes:
        -----------
        variable_reactions: List[bool] 
            Indicates whether each numerical feature is above the selection threshold.
        """
        super().__init__()
        self.numerical_features = numerical_features
        self.selection_threshold = selection_threshold
        self.target_variable = target_variable
        self.variable_reactions = None
        
    def fit(self, X: pd.DataFrame, y = None) -> 'FeatureSelectorStdDev':
        """
        Fit the FeatureSelector to the data by calculating the standard deviations of the features across rows.

        Args:
        ----
            X: pd.DataFrame 
                The input data containing the features.

        Returns:
        -------
            FeatureSelector: The instance itself.
        """
        std_devs = X[self.numerical_features].std()
        threshold = std_devs.quantile(self.selection_threshold)
        self.variable_reactions = std_devs > threshold
        return self
        
    def transform(self, X):
        """
        Transform the data by selecting features.

        Args:
            X (DataFrame): The input data to transform.

        Returns:
            DataFrame: The transformed data with selected features.
        """
        return X.loc[:, self.variable_reactions]
    
