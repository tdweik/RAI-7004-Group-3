import unittest
from src.algorithms.classification import LogisticRegressionModel, DecisionTreeModel, RandomForestModel
from src.algorithms.regression import LinearRegressionModel, DecisionTreeRegressorModel, RandomForestRegressorModel
import pandas as pd

class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.classification_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        self.regression_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [1, 2, 3, 4, 5]
        })

    def test_logistic_regression(self):
        model = LogisticRegressionModel()
        model.train(self.classification_data, 'target')
        predictions = model.predict(self.classification_data[['feature1', 'feature2']])
        self.assertEqual(len(predictions), len(self.classification_data))

    def test_decision_tree_classification(self):
        model = DecisionTreeModel()
        model.train(self.classification_data, 'target')
        predictions = model.predict(self.classification_data[['feature1', 'feature2']])
        self.assertEqual(len(predictions), len(self.classification_data))

    def test_random_forest_classification(self):
        model = RandomForestModel()
        model.train(self.classification_data, 'target')
        predictions = model.predict(self.classification_data[['feature1', 'feature2']])
        self.assertEqual(len(predictions), len(self.classification_data))

    def test_linear_regression(self):
        model = LinearRegressionModel()
        model.train(self.regression_data, 'target')
        predictions = model.predict(self.regression_data[['feature1', 'feature2']])
        self.assertEqual(len(predictions), len(self.regression_data))

    def test_decision_tree_regression(self):
        model = DecisionTreeRegressorModel()
        model.train(self.regression_data, 'target')
        predictions = model.predict(self.regression_data[['feature1', 'feature2']])
        self.assertEqual(len(predictions), len(self.regression_data))

    def test_random_forest_regression(self):
        model = RandomForestRegressorModel()
        model.train(self.regression_data, 'target')
        predictions = model.predict(self.regression_data[['feature1', 'feature2']])
        self.assertEqual(len(predictions), len(self.regression_data))

if __name__ == '__main__':
    unittest.main()