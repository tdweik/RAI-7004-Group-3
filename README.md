# Machine Learning Pipeline CLI

This project provides a command-line interface (CLI) application for experimenting with and developing machine learning pipelines using scikit-learn. It allows users to easily train models on datasets and evaluate their performance.

## Features

- Support for both classification and regression algorithms.
- Ability to load datasets from CSV files.
- Cross-validation for performance evaluation.
- Persistence of trained models as `.pkl` files.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### Command Line Interface

You can run the CLI application using the following command:

```
python -m src.cli <path_to_csv> <target_column> <algorithm_name> [--random_state <value>] [--num_folds <value>]
```

**Parameters:**

- `<csv_file>`: Path to the input CSV file (required).
- `<target_column>`: Name of the target column (required).
- `<algorithm>`: Name of the algorithm to use (required). Supported algorithms:
  - Classification: `logistic_regression`, `decision_tree`, `random_forest`
  - Regression: `linear_regression`, `decision_tree_regressor`, `random_forest_regressor`
- `--random_state`: Random state for reproducibility (optional, default is None).
- `--num_folds`: Number of folds for cross-validation (optional, default is 5).

### Jupyter Notebook

You can also use the application in a Jupyter Notebook by importing the necessary modules and calling the functions directly. Here’s an example:

```python
from src.pipeline import ML_Pipeline

pipeline = ML_Pipeline(csv_file='sample_house_prices.csv', target_column='price', algorithm='linear_regression')
pipeline.run()
```

## File Structure

```
.github/
├── workflows/
│   └── ci.yml
.gitignore
.vscode/
├── launch.json
output/
├── models/
│   └── 
src/
├── __init__.py
├── algorithms/
│   ├── __init__.py
│   └── ...
├── cli.py
└── pipeline.py
tests/
├── __init__.py
├── test_algorithms.py
├── test_cli.py
└── test_pipeline_classification.py
README.md
requirements.txt
sample_binary_classification.csv
sample_house_prices.csv
setup.py
```

## Testing

To run the tests, use the following command:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.