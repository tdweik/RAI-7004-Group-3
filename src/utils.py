def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    import pandas as pd
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

def validate_input_params(target_column: str, algorithm: str, valid_algorithms: list) -> None:
    """Validate input parameters for the machine learning pipeline.

    Args:
        target_column (str): The name of the target column.
        algorithm (str): The name of the algorithm to be used.
        valid_algorithms (list): List of valid algorithm names.

    Raises:
        ValueError: If the target column or algorithm is invalid.
    """
    if target_column not in valid_algorithms:
        raise ValueError(f"Invalid target column: {target_column}")
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}")

def calculate_performance_metrics(y_true, y_pred, metric: str) -> float:
    """Calculate performance metrics based on the specified metric.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        metric (str): The performance metric to calculate.

    Returns:
        float: The calculated performance metric.
    """
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")