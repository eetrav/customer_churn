"""
Example using CustomerChurn library to load and assess pretrained models.

Author: Emily Travinsky
Date: 09/2023
"""

import sys

sys.path.append("../")

# https://stackoverflow.com/questions/48836604/avoiding-pylint-complaints-when-importing-python-packages-from-submodules
#pylint: disable=wrong-import-position
from churn_library import CustomerChurn

customer_churn = CustomerChurn("../data/bank_data.csv")

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

# Load models
RFC_PTH = "./models/rfc_model.pkl"
LR_PTH = "./models/logistic_model.pkl"
customer_churn.load_models(RFC_PTH, LR_PTH)

# Assess models
customer_churn.assess_models()
