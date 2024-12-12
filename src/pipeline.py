class ML_Pipeline:
    def __init__(self, csv_file: str, target_column: str, algorithm: str, random_state: int = 42, num_folds: int = 5):
        """
        Initializes the ML_Pipeline with the provided parameters.

        :param csv_file: Path to the CSV file containing the dataset.
        :param target_column: Name of the target column in the dataset.
        :param algorithm: Name of the algorithm to use for modeling.
        :param random_state: Random state for reproducibility (default is 42).
        :param num_folds: Number of folds for cross-validation (default is 5).
        """
        self.csv_file = csv_file
        self.target_column = target_column
        self.algorithm = algorithm
        self.random_state = random_state
        self.num_folds = num_folds
        self.pipeline = None
        self.model = None

    def load_data(self):
        """
        Loads the dataset from the CSV file.
        """
        import pandas as pd
        self.data = pd.read_csv(self.csv_file)

    def preprocess_data(self):
        """
        Preprocesses the data by separating features and target variable.
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return X, y

    def create_pipeline(self):
        """
        Creates a machine learning pipeline based on the specified algorithm.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
        from algorithms.classification import ClassificationAlgorithms
        from algorithms.regression import RegressionAlgorithms

        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        if self.algorithm in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
            self.model = ClassificationAlgorithms(self.algorithm)
        elif self.algorithm in ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor']:
            self.model = RegressionAlgorithms(self.algorithm)
        else:
            raise ValueError("Unsupported algorithm specified.")

        self.pipeline = Pipeline(steps=[('model', self.model)])

        # Fit the model
        self.pipeline.fit(X_train, y_train)

        # Cross-validated performance metrics
        scores = cross_val_score(self.pipeline, X, y, cv=self.num_folds)
        return scores.mean()

    def save_model(self, filename: str):
        """
        Saves the trained model as a .pkl file.

        :param filename: The name of the file to save the model.
        """
        import joblib
        joblib.dump(self.pipeline, filename)

    def run(self):
        """
        Executes the ML pipeline: loads data, creates the pipeline, and saves the model.
        """
        self.load_data()
        score = self.create_pipeline()
        self.save_model('trained_model.pkl')
        return score