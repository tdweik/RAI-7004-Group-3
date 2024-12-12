from setuptools import setup, find_packages

setup(
    name='ml-pipeline-cli',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A CLI application for experimenting with machine learning pipelines.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'click',
        'scikit-learn',
        'pandas',
        'numpy',
        'joblib'
    ],
    entry_points={
        'console_scripts': [
            'ml-pipeline=cli:main',
        ],
    },
)