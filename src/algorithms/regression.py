class LinearRegressionModel:
    """Class for Linear Regression model."""
    
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

    def train(self, X, y):
        """Train the Linear Regression model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)


class DecisionTreeRegressorModel:
    """Class for Decision Tree Regressor model."""
    
    def __init__(self):
        from sklearn.tree import DecisionTreeRegressor
        self.model = DecisionTreeRegressor()

    def train(self, X, y):
        """Train the Decision Tree Regressor model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)


class RandomForestRegressorModel:
    """Class for Random Forest Regressor model."""
    
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor()

    def train(self, X, y):
        """Train the Random Forest Regressor model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)