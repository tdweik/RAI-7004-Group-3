import pytest
from click.testing import CliRunner
from src.cli import main

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output

def test_cli_with_valid_input():
    runner = CliRunner()
    result = runner.invoke(main, ['data.csv', 'target', 'RandomForestClassifier'])
    assert result.exit_code == 0
    assert 'Model training completed' in result.output

def test_cli_with_invalid_algorithm():
    runner = CliRunner()
    result = runner.invoke(main, ['data.csv', 'target', 'InvalidAlgorithm'])
    assert result.exit_code != 0
    assert 'Invalid algorithm specified' in result.output

def test_cli_with_missing_parameters():
    runner = CliRunner()
    result = runner.invoke(main, ['data.csv'])
    assert result.exit_code != 0
    assert 'Error: Missing parameters' in result.output

def test_cli_with_optional_parameters():
    runner = CliRunner()
    result = runner.invoke(main, ['data.csv', 'target', 'LogisticRegression', '--random_state', '42'])
    assert result.exit_code == 0
    assert 'Model training completed' in result.output