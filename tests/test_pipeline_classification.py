import pytest
import os
import pandas as pd
from sklearn.pipeline import Pipeline

from src.pipeline import ML_Pipeline  # Import your pipeline class

@pytest.fixture
def sample_data(tmpdir):
    """
    Fixture to create a sample dataset for testing.
    """
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    csv_file = tmpdir.join("sample_binary_classification.csv")
    df.to_csv(csv_file, index=False)
    return str(csv_file)

@pytest.fixture
def pipeline_instance(sample_data):
    """
    Fixture to initialize the ML_Pipeline instance.
    """
    return ML_Pipeline(csv_file=sample_data, target_column='target', algorithm='random_forest_classifier')

def test_pipeline_initialization(pipeline_instance, sample_data):
    """
    Test initialization of the pipeline.
    """
    pipeline = pipeline_instance
    assert pipeline.csv_file == sample_data, "The csv_file attribute is incorrect."
    assert pipeline.target_column == 'target', "The target_column attribute is incorrect."
    assert pipeline.algorithm == 'random_forest_classifier', "The algorithm attribute is incorrect."
    assert pipeline.random_state == 42, "Default random_state should be 42."
    assert pipeline.num_folds == 5, "Default number of folds should be 5."

def test_load_data(pipeline_instance):
    """
    Test the load_data method.
    """
    pipeline = pipeline_instance
    pipeline.load_data()
    assert not pipeline.data.empty, "Data should not be empty after loading."
    assert 'feature1' in pipeline.data.columns, "Feature1 should be in the dataset."
    assert 'feature2' in pipeline.data.columns, "Feature2 should be in the dataset."
    assert 'target' in pipeline.data.columns, "Target should be in the dataset."

def test_preprocess_data(pipeline_instance):
    """
    Test the preprocess_data method.
    """
    pipeline = pipeline_instance
    pipeline.load_data()
    pipeline.preprocess_data()

    assert hasattr(pipeline, 'X_train') and len(pipeline.X_train) > 0, "X_train should not be empty."
    assert hasattr(pipeline, 'X_test') and len(pipeline.X_test) > 0, "X_test should not be empty."
    assert hasattr(pipeline, 'y_train') and len(pipeline.y_train) > 0, "y_train should not be empty."
    assert hasattr(pipeline, 'y_test') and len(pipeline.y_test) > 0, "y_test should not be empty."

def test_create_pipeline(pipeline_instance):
    """
    Test the creation of the pipeline.
    """
    pipeline = pipeline_instance
    pipeline.create_pipeline()
    assert isinstance(pipeline.pipeline, Pipeline), "Pipeline should be an instance of sklearn's Pipeline."
    assert pipeline.pipeline is not None, "Pipeline object should not be None."

def test_save_model(pipeline_instance, tmpdir):
    """
    Test the save_model method.
    """
    pipeline = pipeline_instance
    pipeline.create_pipeline()
    pipeline.save_model()

    model_file = os.path.join(pipeline.output_dir, f"random_forest_classifier_{pipeline.unique_id}.pkl")
    print(f"Expected model file: {model_file}")

    assert os.path.exists(model_file), "Model file should be created."


def test_missing_file():
    """
    Test loading data from a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
       ML_Pipeline(csv_file="non_existent_file.csv", target_column='target', algorithm='random_forest_classifier').load_data()
