from typing import Optional
import click
import pandas as pd
from src.pipeline import ML_Pipeline

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('target_column', type=str)
@click.argument('algorithm', type=click.Choice(['logistic_regression', 'decision_tree', 'random_forest', 
                                                  'linear_regression', 'decision_tree_regressor', 
                                                  'random_forest_regressor']))
@click.option('--random_state', default=42, help='Random state for reproducibility.')
@click.option('--num_folds', default=5, help='Number of folds for cross-validation.')
def main(csv_file: str, target_column: str, algorithm: str, random_state: Optional[int], num_folds: Optional[int]):
    """CLI for running machine learning pipelines."""
    click.echo(f'Loading data from {csv_file}...')
    data = pd.read_csv(csv_file)
    
    click.echo(f'Initializing ML Pipeline with target column: {target_column} and algorithm: {algorithm}...')
    pipeline = ML_Pipeline(data, target_column, algorithm, random_state, num_folds)
    
    click.echo('Training model...')
    pipeline.train()
    
    click.echo('Evaluating model...')
    metrics = pipeline.evaluate()
    
    click.echo(f'Model performance metrics: {metrics}')
    
    click.echo('Saving model...')
    pipeline.save_model()

if __name__ == '__main__':
    main()