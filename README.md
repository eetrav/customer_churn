# customer_churn
Customer Churn Project for Udacity Clean Code Principles

This Python module was developed for the final project of the Udacity Clean 
Coding Principles Course, as part of the Machine Learning DevOps nanodegree:
https://learn.udacity.com/nanodegrees/nd0821

The project goal was to take a developmental notebook, churn_notebook.ipynb,
and refactor it into a production-ready Python module that follows coding best
practices, including logging and testing.

Code was developed using Python 3.8, and a requirements_py3.8.txt file is 
provided to assist in setting up a working environment. These requirements can 
be installed with ```python -m pip install -r requirements_py3.6.txt```

The main functionality of the churn_notebook has been moved into the 
CustomerChurn class in churn_library.py. This class can be used to train and 
test a Random Forest and Logistic Regression classifier, or to import
pretrained models and assess their performance with new data. Running ```python
churn_library.py``` creates the ./logs/churn_analysis.log file with added
insights.

Two example scripts are provided in the examples directory, which also provides
example outputs, including logging, exploratory data analysis, resulting
imagery, and exported trained models.

The testing directory includes a testing script that can be run with the Pytest
framework by running ```pytest``` from the main customer_churn directory. Running
```pytest``` will generate a separate logging script stored in ./tests/logs/
which includes results on the tests pass/fail status. The testing directory 
includes test_data which is used in some of the unit tests.