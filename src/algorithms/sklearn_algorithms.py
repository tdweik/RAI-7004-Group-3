from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class SklearnAlgorithm:
    """
    A class to encapsulate various scikit-learn algorithms for both 
    classification and regression tasks.
    Attributes:
    -----------
    algorithm_name : str
        The name of the algorithm to be used.
    model : object
        The scikit-learn model instance.
    Methods:
    --------
    __init__(algorithm_name: str):
        Initializes the SklearnAlgorithm with the specified algorithm name.
    fit(X, y):
        Fits the model to the provided training data.
    predict(X):
        Predicts the target values for the provided input data.
    get_model():
        Returns the trained model instance.
    save_model(file_path: str):
        Saves the trained model to the specified file path using joblib.
    score(X, y):
        Returns the score of the model on the provided test data.
    """
    def __init__(self, algorithm_name: str):
        self.model = None
        self.algorithm_name = algorithm_name
        if self.algorithm_name == 'logistic_regression':
            self.model = LogisticRegression()
        elif self.algorithm_name == 'decision_tree_classifier':
            self.model = DecisionTreeClassifier()
        elif self.algorithm_name == 'random_forest_classifier':
            self.model = RandomForestClassifier()
        elif self.algorithm_name == 'linear_regression':
            self.model = LinearRegression()
        elif self.algorithm_name == 'decision_tree_regressor':
            self.model = DecisionTreeRegressor()
        elif self.algorithm_name == 'random_forest_regressor':
            self.model = RandomForestRegressor()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns:
        self : object
            Returns self.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input data to predict.

        Returns:
        --------
        array-like of shape (n_samples,)
            The predicted target values.

        Raises:
        -------
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the machine learning model instance.

        Returns:
            object: The machine learning model instance.
        """
        return self.model

    def save_model(self, file_path: str):
        """
        Save the trained model to a file.

        Parameters:
        file_path (str): The path where the model will be saved.

        Returns:
        None
        """
        import joblib
        joblib.dump(self.model, file_path)
        
    def score(self, X, y):
        """
        Evaluate the performance of the trained model on the given test data and labels.

        Parameters:
        X (array-like): Test data.
        y (array-like): True labels for the test data.

        Returns:
        float: The score of the model on the given test data and labels.

        Raises:
        ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.score(X, y)