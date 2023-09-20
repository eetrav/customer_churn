"""
Example code for how to use CustomerChurn library to train and assess models.

Author: Emily Travinsky
Date: 09/2023
"""

import sys

sys.path.append("../")

# https://stackoverflow.com/questions/48836604/avoiding-pylint-complaints-when-importing-python-packages-from-submodules
#pylint: disable=wrong-import-position
from churn_library import CustomerChurn

# Instantiate CustomerChurn class with valid bank data
customer_churn = CustomerChurn("../data/bank_data.csv")

# Perform Exploratory Data Analysis
plotting_dict = {'hist': ['Churn', 'Customer_Age'],
                 'counts': ['Marital_Status'],
                 'dist': ['Total_Trans_Ct'],
                 'corr': True}
customer_churn.perform_eda(plotting_dict)

# Use encoder_helper to update bank_data - relate feature columns to churn
categories_2_encode = ['Gender', 'Education_Level', 'Marital_Status',
                       'Income_Category', 'Card_Category']
ENCODED_RESPONSE_STR = 'Churn'
customer_churn.encoder_helper(categories_2_encode,
                              ENCODED_RESPONSE_STR)

# Create train and test data
customer_churn.perform_feature_engineering(ENCODED_RESPONSE_STR)

# Train models
customer_churn.train_models()

# Assess models
customer_churn.assess_models()

# Save Models
customer_churn.export_models()
