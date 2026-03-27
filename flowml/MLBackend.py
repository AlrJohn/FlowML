import pandas as pd
# Data manipulation
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.base import is_regressor

# Train/test splitting
from sklearn.model_selection import train_test_split

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Clustering models
from sklearn.cluster import KMeans


MODEL_MAP = {
            'LogisticRegression': LogisticRegression,
            'RandomForest':       RandomForestClassifier,
            'SVM':                SVC,
            'KNN':                KNeighborsClassifier,
            'NaiveBayes':         GaussianNB,
            'LinearRegression':   LinearRegression,
            'Ridge':              Ridge,
            'Lasso':              Lasso,
            'KMeans':             KMeans,
        }

PARAM_MAP = {
    'trees':     'n_estimators',
    'neighbors': 'n_neighbors',
    'clusters':  'n_clusters',
    'alpha':     'alpha',
}
class MLBackend:
    def __init__(self):
        self.df = None  # This will hold the active DataFrame
        self.train_set = None  # This will hold the training set after a split
        self.test_set = None  # This will hold the test set after a split
        self.model = None  # This will hold the trained model
        self.normalization_columns = []
        self._target_encoding_map = None



    def load(self, filename: str) -> pd.DataFrame:
        self.df = pd.read_csv(filename)
        return self.df
    
    def drop(self, column_names: list) -> pd.DataFrame:
        if self.df is None:
            raise Exception("MLBackend: No active DataFrame to drop columns from")
        for column_name in column_names:
            if column_name not in self.df.columns:
                raise Exception(f"MLBackend: Column '{column_name}' does not exist in the active DataFrame")
        
        self.df = self.df.drop(columns=column_names)
        return self.df
    
    def normalize(self, column_names: list) -> pd.DataFrame:
        if self.df is None:
            raise Exception("MLBackend: No active DataFrame to normalize")
        for column_name in column_names:
             if column_name not in self.df.columns:
                raise Exception(f"MLBackend: Column '{column_name}' does not exist in the active DataFrame")
        
        self.normalization_columns = column_names
        return self.df  
    
    def split(self, train: float, test: float) -> tuple:
        if self.df is None:
            raise Exception("MLBackend: No active DataFrame to split")
        if not (train + test == 1.0):
            raise Exception("MLBackend: Train and test ratios must sum to 1")
        
        
        X = self.df.iloc[:, :-1]  # all columns except the last one as features
        y = self.df.iloc[:, -1]   # the last column as the target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)
        
        if self.normalization_columns:
            scaler = StandardScaler()
            X_train[self.normalization_columns] = scaler.fit_transform(X_train[self.normalization_columns])
            X_test[self.normalization_columns] = scaler.transform(X_test[self.normalization_columns])

            self.normalization_columns = []  # reset normalization columns after applying normalization

        self.train_set = (X_train, y_train)
        self.test_set = (X_test, y_test)
        return self.train_set, self.test_set


    def set_model(self, model_name: str, params: dict = None):
        if model_name not in MODEL_MAP:
            raise Exception(f"MLBackend: Unknown model type '{model_name}'")
        # translate FlowML param names to sklearn param names
        sklearn_params = {}
        for key, val in (params or {}).items():
            sklearn_key = PARAM_MAP.get(key, key)
            sklearn_params[sklearn_key] = val
        self.model = MODEL_MAP[model_name](**sklearn_params)
        self._target_encoding_map = None

    def train(self, dataset: tuple):
        if self.model is None:
            raise Exception("MLBackend: No model defined. Call 'model' before 'train'.")
        X, y = dataset
        # Regression estimators require numeric targets. If labels are categorical
        # strings, encode them to stable integer ids for fit/evaluate.
        if is_regressor(self.model) and not pd.api.types.is_numeric_dtype(y):
            unique_labels = pd.unique(y)
            self._target_encoding_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = y.map(self._target_encoding_map).astype(float)
        self.model.fit(X, y)

    def evaluate(self, dataset: tuple) -> float:
        if self.model is None:
            raise Exception("MLBackend: No model defined.")
        X, y = dataset
        if self._target_encoding_map is not None:
            y = y.map(self._target_encoding_map).fillna(-1).astype(float)
        return self.model.score(X, y)
