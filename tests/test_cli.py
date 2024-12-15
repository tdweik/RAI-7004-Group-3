import sys
import subprocess
import pytest

sys.path.append('src')


def test_cli_command():
    result = subprocess.run(
        ['python', 'src/cli.py', 'some_command'], capture_output=True, text=True
    )
    assert 'Error' in result.stderr


def test_cli_invalid_command():
    result = subprocess.run(
        ['python', 'src/cli.py', 'invalid_command'], capture_output=True, text=True
    )
    assert 'Error' in result.stderr


def test_cli_with_invalid_algorithm():
    csv_file = 'sample_binary_classification.csv'
    target_column = 'target'
    algorithm = 'invalid_algorithm'
    result = subprocess.run(
        ['python', 'src/cli.py', csv_file, target_column, algorithm], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert 'Error' in result.stderr


if __name__ == '__main__':
    pytest.main()