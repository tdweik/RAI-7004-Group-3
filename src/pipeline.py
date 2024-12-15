import sys
sys.path.append('src')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from algorithms.sklearn_algorithms import SklearnAlgorithm
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report
import pandas as pd
import os
import joblib
import uuid
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ML_Pipeline:
    def __init__(self, csv_file: str, target_column: str, algorithm: str, 
                 random_state: int = 42, num_folds: int = 5):
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
        self.unique_id = uuid.uuid4()
        
        # Create the output models directory if it doesn't exist
        self.output_dir = os.path.join(os.getcwd(), 'output', 'models')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Loads the dataset from the CSV file.
        """

        self.data = pd.read_csv(self.csv_file)

    def preprocess_data(self):
        """
        Preprocesses the data by separating features and target variable.
        """
        self.load_data()
        
        self.data = self.data.drop_duplicates()

        # Handle missing values (example: fill with mean for numeric columns)
        for column in self.data.select_dtypes(include=['number']).columns:
            self.data[column] = self.data[column].fillna(self.data[column].mean())

        # Handle missing values (example: fill with mode for categorical columns)
        for column in self.data.select_dtypes(include=['object']).columns:
            self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
            
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
    def create_pipeline(self):
        """
        Creates a machine learning pipeline based on the specified algorithm.
        """
        self.preprocess_data()
        
        # Identify categorical and numerical columns
        categorical_features = self.X.select_dtypes(
            include=['object', 'category']
        ).columns
        numerical_features = self.X.select_dtypes(
            include=['int64', 'float64']
        ).columns

        # Create the column transformer with OneHotEncoder and MinMaxScaler
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        self.model = SklearnAlgorithm(self.algorithm)
        
        self.pipeline = Pipeline(steps=[('model', self.model)])

        # Update the pipeline to include the preprocessor
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])

        # Fit the model
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluates the model using cross-validation and returns the performance metrics.
        """

        # Cross-validated performance metrics
        scores = cross_val_score(
            self.pipeline, self.X, self.y, cv=self.num_folds
        )
        print(f"Cross-validated scores: {scores}")
        print(f"Mean cross-validated score: {scores.mean()}")

        # Evaluate on test set
        y_pred = self.pipeline.predict(self.X_test)

        if self.algorithm in ['logistic_regression', 'random_forest']:
            # Classification metrics
            accuracy = round(accuracy_score(self.y_test, y_pred), 2)
            print(f"Accuracy on test set: {accuracy}")

            # Generate classification report
            class_report = classification_report(self.y_test, y_pred)
            print(f"Classification Report:\n{class_report}")

            # Export results to a text file
            with open(f"{self.output_dir}/{self.algorithm}_{self.unique_id}_evaluation.txt", "w") as f:
                f.write(f"Cross-validated scores: {scores}\n")
                f.write(f"Mean cross-validated score: {scores.mean()}\n")
                f.write(f"Accuracy on test set: {accuracy}\n")
                f.write(f"Classification Report:\n{class_report}\n")

        elif self.algorithm in ['linear_regression', 'random_forest_regression']:
            # Regression metrics
            mae = round(mean_absolute_error(self.y_test, y_pred), 2)
            mse = round(mean_squared_error(self.y_test, y_pred), 2)
            print(f"Mean Absolute Error on test set: {mae}")
            print(f"Mean Squared Error on test set: {mse}")

            # Export results to a text file
            with open(f"{self.output_dir}/{self.algorithm}_{self.unique_id}_evaluation.txt", "w") as f:
                f.write(f"Cross-validated scores: {scores}\n")
                f.write(f"Mean cross-validated score: {scores.mean()}\n")
                f.write(f"Mean Absolute Error on test set: {mae}\n")
                f.write(f"Mean Squared Error on test set: {mse}\n")
        return

    def save_model(self):
        """
        Saves the trained model as a .pkl file.
        """
        # Generate a common UUID for the evaluation and model files
       
        filename = f"{self.output_dir}/{self.algorithm}_{self.unique_id}"
        joblib.dump(self.pipeline, f"{filename}.pkl")
        
    def run(self):
        """
        Runs the ML pipeline by creating the pipeline, evaluating the model, and saving it.
        """
        self.create_pipeline()
        self.evaluate()
        self.save_model()