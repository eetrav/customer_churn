"""
Testing for churn_library.py functionality.

This module will test whether appropriate errors are raised under qualifying
conditions (generally for invalid user input data), and that data engineering is
successfully performed and output files are generated.

Author: Emily Travinsky
Date: 09/2023
"""

import logging
import os
import shutil
import sys

sys.path.append("./")

# https://stackoverflow.com/questions/48836604/avoiding-pylint-complaints-when-importing-python-packages-from-submodules
#pylint: disable=wrong-import-position
import pytest
from churn_library import CustomerChurn


if not os.path.exists('./tests/logs/'):
    os.makedirs('./tests/logs/')

if not os.path.exists('./tests/output/'):
    os.makedirs('./tests/output/')

# Create a separate testing log from the churn_library training log
# https://stackoverflow.com/questions/58020019/how-to-separate-log-handlers-in-python
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.FileHandler('./tests/logs/churn_library_testing.log', 'w+')
handler.setFormatter(formatter)

testing_logger = logging.getLogger('test_logger')
testing_logger.setLevel(logging.INFO)
testing_logger.addHandler(handler)


@pytest.fixture(name="valid_churn")
def fixture_valid_churn() -> CustomerChurn:
    """
    Pytest fixture of valid bank data to use in testing downstream functions.

    Returns:
        CustomerChurn: Instance of CustomerChurn class with valid bank data.
    """
    churn = CustomerChurn(
        "./tests/test_data/bank_data.csv",
        image_output=os.path.join(
            "tests",
            "output",
            "images"))
    return churn


@pytest.fixture(name="dropped_cols")
def fixture_dropped_cols() -> CustomerChurn:
    """
    Pytest fixture of invalid bank data to use in testing downstream functions.

    This fixture creates an instance of the CustomerChurn class where certain
    necessary columns are missing, on order to check that errors are returned
    when requesting certain functionality (EDA, encoding, feature engineering).

    Returns:
        CustomerChurn: Instance of CustomerChurn class where certain bank data
                       variables are missing.
    """
    churn = CustomerChurn(
        "./tests/test_data/dropped_columns_eda.csv",
        image_output=os.path.join(
            "tests",
            "output",
            "images"))
    return churn


@pytest.fixture(name="churn_assess_models")
def fixture_churn_assess_models() -> CustomerChurn:
    """
    Pytest fixture that loads and encodes valid bank data.

    This fixture is used to test that errors are raised when the user requests
    the assess_models function without
    1) instantiating train/test data, or
    2) training/loading valid models first

    Returns:
        CustomerChurn: Instance of CustomerChurn class with valid bank data that
            has encoding applied, but no train/test data or valid models.
    """

    churn = CustomerChurn(
        "./tests/test_data/bank_data.csv",
        image_output=os.path.join(
            "tests",
            "output",
            "images"))

    # Use encoder_helper to update bank_data - relate feature columns to churn
    categories_2_encode = ['Gender', 'Education_Level', 'Marital_Status',
                           'Income_Category', 'Card_Category']
    encoded_response_str = 'Churn'
    churn.encoder_helper(categories_2_encode,
                         encoded_response_str)
    return churn


@pytest.fixture(name="churn_pretrained_models")
def fixture_churn_pretrained_models() -> CustomerChurn:
    """
    Pytest fixture that skips model training to test downstream functionality.

    This fixture imports pretrained models to test the assess_models and
    export_models functionality, allowing the testing framework to bypass
    EDA and model training.

    Returns:
        CustomerChurn: Instance of CustomerChurn class with valid bank data that
            has train/test data and imported pre-trained models.
    """
    # Remove all files generated during previous testing.
    shutil.rmtree("./tests/output/")

    # Create new CustomerChurn instance with output test directory.
    churn = CustomerChurn(
        "./tests/test_data/bank_data.csv",
        image_output=os.path.join(
            "tests",
            "output",
            "images"),
        model_output=os.path.join(
            "tests",
            "output",
            "models"))
    categories_2_encode = ['Gender', 'Education_Level', 'Marital_Status',
                           'Income_Category', 'Card_Category']
    encoded_response_str = 'Churn'
    churn.encoder_helper(categories_2_encode,
                         encoded_response_str)
    # Generate train/test data to use in analysis of pre-trained models
    churn.perform_feature_engineering(encoded_response_str)
    # Load pre-trained models to test model analysis
    rfc_pth = "./tests/pre-trained_models/rfc_model.pkl"
    lr_pth = "./tests/pre-trained_models/logistic_model.pkl"
    churn.load_models(rfc_pth, lr_pth)
    return churn


def test_instantiate_invalid_file_raises_err():
    """
    Test that FileNotFoundError occurs when CSV does not exist.
    """
    testing_logger.info("""Checking that FileNotFoundError occurrs when
                        instantiating CustomerChurn with non-existing file.""")
    with pytest.raises(FileNotFoundError):
        CustomerChurn("./tests/test_data/bank_data2.csv")


def test_instantiate_empty_df_raises_err():
    """
    Test that empty dataframe produces AssertionError.
    """
    testing_logger.info("""Checking that AssertionError is raised for empty
                        dataframe.""")
    with pytest.raises(AssertionError):
        CustomerChurn("./tests/test_data/empty_df.csv")


def test_create_churn_col_without_attrition():
    """
    Test that KeyError occurs if Attrition_Flag is missing from bank data.
    """
    testing_logger.info(
        """Testing that 'Churn' column cannot be created without
                        an Attrition_Flag in dataframe.""")
    with pytest.raises(KeyError):
        CustomerChurn("./tests/test_data/df_no_attrition_flag.csv")


def test_create_churn_col(valid_churn: CustomerChurn):
    """
    Test that the 'Churn' column was successfully created in CustomerChurn.

    Args:
        valid_churn (CustomerChurn): Pytest fixture instance of CustomerChurn
                                     class with valid bank data.
    """
    testing_logger.info("""Testing that 'Churn' column can be created when
                        dataframe has valid Attrition_Flag column.""")
    assert 'Churn' in valid_churn.bank_data.columns


def test_eda_with_missing_cols_raises_err(dropped_cols: CustomerChurn):
    """
    Test to check that EDA returns error if requested columns are missing.

    Args:
        dropped_cols (CustomerChurn): Pytest fixture instance of CustomerChurn
                                      class with missing columns.
    """
    testing_logger.info("""Testing that Exploratory Data Analysis returns error
                        when missing columns are requested.""")
    plot_dict = {'hist': ['Churn', 'Customer_Age'],
                 'counts': ['Marital_Status'],
                 'dist': ['Total_Trans_Ct'],
                 'corr': True}
    with pytest.raises(KeyError):
        dropped_cols.perform_eda(plot_dict)


def test_eda_produces_images(valid_churn: CustomerChurn):
    """
    Test EDA outputs are successfully generated with valid bank data.

    Args:
        valid_churn (CustomerChurn): Pytest fixture instance of CustomerChurn
                                     class with valid bank data.
    """
    testing_logger.info("""Testing that Exploratory Data Analysis produces
                        correct images when valid dataframe is provided.""")
    plotting_dict = {'hist': ['Churn', 'Customer_Age'],
                     'counts': ['Marital_Status'],
                     'dist': ['Total_Trans_Ct'],
                     'corr': True}
    valid_churn.perform_eda(plotting_dict)
    images = [
        "Churn_hist.png",
        "Correlation.png",
        "Customer_Age_hist.png",
        "Marital_Status_counts.png",
        "Total_Trans_Ct_dist.png"]
    assert all(fname in os.listdir("./tests/output/images/eda/")
               for fname in images)


def test_encoding_missing_col_raises_err(dropped_cols: CustomerChurn):
    """
    Test to check that encoding returns error if requested columns are missing.

    Args:
        dropped_cols (CustomerChurn): Pytest fixture instance of CustomerChurn
                                      class with missing columns.
    """
    testing_logger.info("""Testing for raised AssertionError when missing
                        column is requested for encoding.""")
    with pytest.raises(AssertionError):
        dropped_cols.encoder_helper(category_lst=['Customer_Age'])


def test_encoding_cols(valid_churn: CustomerChurn):
    """
    Test using results of churn_trained_models fixture to check column encoding.

    Args:
        valid_churn (CustomerChurn): Pytest fixture instance of CustomerChurn
                                     class with valid bank data.
    """
    testing_logger.info("""Testing that valud columns will be encoded.""")
    categories = ['Gender']
    valid_churn.encoder_helper(categories, 'Churn')
    assert 'Gender_Churn' in valid_churn.bank_data.columns


def test_perform_feature_engineering_missing_response(
        valid_churn: CustomerChurn):
    """
    Test to check error with invalid request for feature engineering.

    Args:
        valid_churn (CustomerChurn): Pytest fixture instance of CustomerChurn
                                     class with valid bank data.
    """
    testing_logger.info("""Testing that KeyError is raised when feature
                        engineering is performed on invalid column.""")
    categories_2_encode = ['Gender', 'Education_Level', 'Marital_Status',
                           'Income_Category', 'Card_Category']
    encoded_response_str = 'Churn'
    valid_churn.encoder_helper(categories_2_encode,
                               encoded_response_str)
    with pytest.raises(KeyError):
        valid_churn.perform_feature_engineering("false_response")


def test_assess_models_without_data(churn_assess_models: CustomerChurn):
    """
    Test that KeyError is raised when assessing models without train/test data.

    Args:
        churn_assess_models (CustomerChurn): Pytest fixture instance of
                                             CustomerChurn class without valid
                                             train/test data or models.
    """
    testing_logger.info("""Testing that KeyError is raised when running
                        assess_models method without valid train/test data.""")
    # Import valid models
    rfc_pth = "./tests/models/rfc_model.pkl"
    lr_pth = "./tests/models/logistic_model.pkl"
    churn_assess_models.load_models(rfc_pth, lr_pth)
    # Assess models without performing feature_engineering to generate test
    # data
    with pytest.raises(KeyError):
        churn_assess_models.assess_models()


def test_assess_models_without_models(churn_assess_models: CustomerChurn):
    """
    Test that AttributeError is raised when assessing NoneType models.

    Args:
        churn_assess_models (CustomerChurn): Pytest fixture instance of
                                             CustomerChurn class without valid
                                             train/test data or models.
    """
    testing_logger.info("""Testing that AttributeError is raised when running
                        assess_models method without valid models.""")
    # Create valid train/test data
    churn_assess_models.perform_feature_engineering('Churn')
    # Assess models without training or importing
    with pytest.raises(AttributeError):
        churn_assess_models.assess_models()


def test_class_reports(churn_pretrained_models: CustomerChurn):
    """
    Test to check classification report outputs of churn_trained_models fixture.

    Args:
        churn_pretrained_models (CustomerChurn): Pytest fixture instance of
                                                 CustomerChurn class imported
                                                 pretrained models.
    """
    churn_pretrained_models.assess_models()
    testing_logger.info("""Testing that class reports are successfully
                        generated after model training.""")
    assert all(fname in os.listdir(os.path.join("./tests/output/images/results"))
               for fname in ["Logistic_Regression_class_report.png",
                             "Random_Forest_class_report.png"])


def test_model_roc_curves(churn_pretrained_models):
    """
    Test to check ROC curve outputs of churn_trained_models fixture.

    Args:
        churn_pretrained_models (CustomerChurn): Pytest fixture instance of
                                                 CustomerChurn class imported
                                                 pretrained models.
    """
    churn_pretrained_models.assess_models()
    testing_logger.info("""Testing that ROC curve plots are successfully
                        created after model training.""")
    assert os.path.exists(
        "./tests/output/images/results/Model_Comparison_ROC.png")


def test_assess_random_forest(churn_pretrained_models):
    """
    Test to check feature importance output of churn_trained_models fixture.

    Args:
        churn_pretrained_models (CustomerChurn): Pytest fixture instance of
                                                 CustomerChurn class imported
                                                 pretrained models.
    """
    churn_pretrained_models.assess_models()
    testing_logger.info("""Testing that analysis plots are successfully
                         created after training Random Forest model.""")
    assert all(fname in os.listdir(os.path.join("./tests/output/images/results"))
               for fname in ["Random_Forest_Feature_Importances.png",
                             "Random_Forest_Tree_Explainer.png"])


def test_save_models(churn_pretrained_models):
    """
    Test to check saved model outputs of churn_trained_models fixture.

    Args:
        churn_pretrained_models (CustomerChurn): Pytest fixture instance of
                                                 CustomerChurn class imported
                                                 pretrained models.
    """
    churn_pretrained_models.export_models()
    testing_logger.info("""Testing that models are saved after training.""")
    assert all(fname in os.listdir("./tests/output/models/")
               for fname in ["rfc_model.pkl", "logistic_model.pkl"])
