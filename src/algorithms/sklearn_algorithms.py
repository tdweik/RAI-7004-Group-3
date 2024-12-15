class SklearnAlgorithm:
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.model = None

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor

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
        
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_model(self):
        return self.model

    def save_model(self, file_path: str):
        import joblib
        joblib.dump(self.model, file_path)
        
    def score(self, X, y):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.score(X, y)