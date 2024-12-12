import unittest
import pandas as pd
from src.pipeline import ML_Pipeline

class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        self.csv_path = 'test_data.csv'
        self.data.to_csv(self.csv_path, index=False)

        self.pipeline = ML_Pipeline(csv_path=self.csv_path, target_column='target', algorithm='Logistic Regression')

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline)

    def test_data_loading(self):
        self.pipeline.load_data()
        self.assertEqual(len(self.pipeline.data), 5)

    def test_model_training(self):
        self.pipeline.load_data()
        self.pipeline.train_model()
        self.assertIsNotNone(self.pipeline.model)

    def test_model_prediction(self):
        self.pipeline.load_data()
        self.pipeline.train_model()
        predictions = self.pipeline.predict()
        self.assertEqual(len(predictions), 5)

    def tearDown(self):
        import os
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

if __name__ == '__main__':
    unittest.main()