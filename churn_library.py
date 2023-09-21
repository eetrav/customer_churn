"""
Class to track bank customers likely to churn.

Author: Emily Travinsky
Date: 09/2023
"""

import logging
import os
from typing import TypedDict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from plotters import Plotters

#os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()

EDAdict = TypedDict('EDAdict', {'hist': list, 'counts': list,
                                'dist': list, 'corr': bool})


class CustomerChurn():
    """
    Class to identify bank members with high churn probability.

    Uses bank data to create models investigating the likelihood of customer
    churn.
    """

    def __init__(
            self,
            bank_data_csv: str = "./data/bank_data.csv",
            image_output: str = "./images",
            model_output: str = "./models"):
        """
        Instantiation of CustomerChurn class with bank data and paths.

        Args:
            bank_data_csv (str, optional): CSV filepath of bank user data.
                Defaults to "./data/bank_data.csv".
            image_output (str, optional): Path to image output folder. Defaults
                to "./images".
            model_output (str, optional): Path to model output folder. Defaults
                to "./models".
        """
        self._create_log()
        self.bank_data: pd.DataFrame = pd.DataFrame()
        self._import_data(bank_data_csv)
        self._create_churn_column()
        # Set default image and model paths, but don't create until necessary
        self.image_pth = image_output
        self.model_pth = model_output

        # Define instance variables for future model training
        self.feature_cols: pd.DataFrame = pd.DataFrame()
        self.data: dict = {}
        self.cv_rfc = None
        self.lrc = None

    def _check_path(self, pth: str = ''):
        """
        Helper function to create path if it does not exist.

        Args:
            pth (str, optional): File path to check. Defaults to ''.
        """

        if not os.path.exists(pth):
            os.makedirs(pth)
            logging.info("SUCCESS: Created %s directory", pth)

    def _create_log(self):
        """
        Function to create logging file for churn investigation.
        """
        self._check_path('./logs/')
        logging.basicConfig(
            filename=os.path.join('./logs', 'churn_analysis.log'),
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')

    def _import_data(self, pth: str = ''):
        """
        Return dataframe for the csv found at input pth.

        Args:
            pth (str, optional): Path to CSV of bank customer data. Defaults to
                                 ''.

        Raises:
            err: FileNotFoundError if input file path not found.
            err: AssertionError if dataframe is empty.
        """

        try:
            self.bank_data = pd.read_csv(pth)
            logging.info("SUCCESS: File imported into dataframe.")
        except FileNotFoundError as err:
            print("Invalid dataframe path.")
            logging.error("ERROR: File path not found.")
            raise err

        if self.bank_data.isnull().values.any():
            warning_text = "NAN value found in dataframe. \
                This may cause errors during exploratory data analysis."
            logging.warning(
                "WARNING: %s", warning_text)

        try:
            assert self.bank_data.shape[0] > 0
            assert self.bank_data.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "ERROR: The file doesn't appear to have rows or columns.")
            raise err

    def _create_churn_column(self):
        """
        Add a T/F customer churn column to bank data, based on Attrition_Flag.

        Raises:
            err: KeyError if Attrition_Flag does not exist in bank data.
        """
        try:
            self.bank_data['Churn'] = self.bank_data['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1)
            logging.info(
                "INFO: Created Churn column based on Attrition_Flag.")
        except KeyError as err:
            logging.error(
                """ERROR: Attrition_Flag must exist in bank data in order to
                create Churn column.""")
            raise err

    def perform_eda(self, plot_dict: EDAdict):
        """
        Function to perform EDA.

        Performs exploratory data analysis on customer data and save figures to
        images folder.

        Args:
            plot_dict (EDAdict): Dictionary of requested plots and dataframe
                                 columns to plot.

        Raises:
            err: KeyError if column does not exist in dataframe for plotting
                 histograms.
            err: KeyError if column does not exist in dataframe for plotting
                 counts.
            err: KeyError if column does not exist in dataframe for plotting
                 distribution.
            err: ValueError if dataframe is empty and cannot plot correlation.
        """

        # If producing EDA output images, check that path exists and/or create
        eda_pth = os.path.join(self.image_pth, 'eda')
        self._check_path(eda_pth)
        plotter = Plotters(self.bank_data, os.path.join(eda_pth))

        for col in plot_dict.get('hist', []):
            try:
                plotter.plot_hist(col)
                logging.info("SUCCESS: Plotted histogram of %s data.", col)
            except KeyError as err:
                logging.warning(
                    "WARNING: Unable to plot histogram of %s in customer bank data.", col)
                raise err

        for col in plot_dict.get('counts', []):
            try:
                plotter.plot_counts(col)
                logging.info("SUCCESS: Plotted counts of %s data.", col)
            except KeyError as err:
                logging.warning(
                    "WARNING: Unable to plot counts of %s in customer bank data.", col)
                raise err

        for col in plot_dict.get('dist', []):
            try:
                plotter.plot_dist(col)
                logging.info("SUCCESS: Plotted distribution of %s data.", col)
            except KeyError as err:
                logging.warning(
                    "WARNING: Unable to plot distribution of %s in customer bank data.", col)
                raise err

        if plot_dict.get('corr', False):
            try:
                plotter.plot_corr()
                logging.info("SUCCESS: Plotted correlation of bank data.")
            except ValueError as err:
                logging.warning(
                    "WARNING: Unable to plot correlation of customer bank data.")
                raise err

    def encoder_helper(self, category_lst: list, response: str = ''):
        """
        Function to encode user-defined categories by churn proportion.

        Turn each categorical column into a new column with propotion of churn
        for each category - associated with cell 15 from the demo notebook.

        Args:
            category_lst (list): List of dataframe columns to compute
                churn proportion.
            response (str, optional): string of response name; optional argument
                that could be used for naming variables of indexing y column.
                Defaults to ''.

        Raises:
            err: AssertionError if requested column is not found in bank data.
        """

        for category in category_lst:
            try:
                assert category in self.bank_data.columns
                cat_groups = self.bank_data.groupby(category).mean()['Churn']
                self.bank_data[category + '_' + \
                    response] = self.bank_data[category].map(cat_groups)
            except AssertionError as err:
                logging.error(
                    "ERROR: %s column not found in bank data.", category)
                raise err

        logging.info(
            "SUCCESS: Created churn encodings based on customer features.")

    def perform_feature_engineering(self, response: str = ''):
        """
        Perform feature engineering to create train and test data.

        Args:
            response (str, optional): response name [optional argument that
                could be used for naming variables or index y column]. Defaults
                to ''.

        Raises:
            err: KeyError if any necessary columns are missing from bank data.
            err: KeyError if user-defined response is not present in bank data.
        """

        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']
        try:
            self.feature_cols = self.bank_data[keep_cols]
        except KeyError as err:
            logging.error(
                "ERROR: %s column(s) missing from dataframe.",
                err.args)
            raise err

        try:
            dependent_response = self.bank_data[response]
        except KeyError as err:
            logging.error("ERROR: %s column not in dataframe.", err.args)
            raise err

        x_train, x_test, y_train, y_test = train_test_split(
            self.feature_cols, dependent_response, test_size=0.3, random_state=42)
        logging.info("SUCCESS: Created independent/dependent train/test sets.")

        self.data = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test}

    def train_models(self):
        """
        Train Random Forest and Logistic Regression models.
        """

        # Random Forest Classifier Training
        rfc = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        self.cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        try:
            self.cv_rfc.fit(self.data['x_train'], self.data['y_train'])
        except KeyError as err:
            logging.error("""ERROR: Training data does not exist! Please run
                          perform_feature_engineering_function.""")
            raise err

        logging.info("SUCCESS: Trained Random Forest Classifier.")

        # Logistic Regression Classifier Training
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        self.lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        try:
            self.lrc.fit(self.data['x_train'], self.data['y_train'])
        except KeyError as err:
            logging.error("""ERROR: Training data does not exist! Please run
                          perform_feature_engineering_function.""")
            raise err

        logging.info("SUCCESS: Trained Logistic Regression Classifier.")

    def _log_and_plot_classification_report(
            self, model_name: str, train_report: str, test_report: str):
        """
        Function to log and plot classification report.

        Args:
            model_name (str): Text to use in report titles and filename.
            train_report (str): Classification report of training data.
            test_report (str): Classification report of testing data.
        """

        logging.info("%s Train", model_name)
        logging.info(train_report)
        logging.info("%s Test", model_name)
        logging.info(test_report)

        plt.figure(figsize=(7, 5))
        plt.text(0.01, 1.05, model_name + " Train",
                 {'fontsize': 14}, fontproperties='monospace')
        plt.text(
            0.01, 0.6, train_report, {
                'fontsize': 12}, fontproperties='monospace')
        plt.text(0.01, 0.45, model_name + " Test",
                 {'fontsize': 14}, fontproperties='monospace')
        plt.text(
            0.01, 0.0, test_report, {
                'fontsize': 12}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            os.path.join(
                self.image_pth, 'results',
                model_name.replace(
                    " ",
                    "_") +
                "_class_report.png"))
        plt.close()

    def _generate_classification_report(
            self,
            y_train_preds: np.ndarray,
            y_test_preds: np.ndarray,
            report_title: str):
        """
        Produces classification report for training and testing data.

        Args:
            y_train_preds (np.ndarray): Predictions for training data.
            y_test_preds (np.ndarray): Predictions for testing data.
            report_title (str): Title of model for classification report.
        """

        train_report = classification_report(
            self.data['y_train'], y_train_preds)
        test_report = classification_report(
            self.data['y_test'], y_test_preds)
        self._log_and_plot_classification_report(
            report_title, train_report, test_report)

    def _plot_roc_curves(self):
        """
        Function to plot ROC curves and compare prediction models.
        """

        plt.figure(figsize=(15, 8))
        plt_ax = plt.gca()
        plot_roc_curve(
            self.cv_rfc.best_estimator_,
            self.data['x_test'],
            self.data['y_test'],
            ax=plt_ax,
            alpha=0.8,
            linewidth=3)
        plot_roc_curve(
            self.lrc,
            self.data['x_test'],
            self.data['y_test'],
            ax=plt_ax,
            alpha=0.8,
            linewidth=3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("False Positive Rate", fontsize=18)
        plt.ylabel("True Positive Rate", fontsize=18)
        plt.legend(prop={'size': 18})
        plt.title("ROC Curve Model Comparison", fontsize=20)
        plt.savefig(
            os.path.join(
                self.image_pth,
                'results',
                "Model_Comparison_ROC.png"))
        plt.close()

    def _assess_random_forest(self):
        """
        Function to create and store the feature importances/explainer.
        """

        plt.figure(figsize=(20, 16))
        plt.title("Tree Explainer for Random Forest", fontsize=20)
        explainer = shap.TreeExplainer(self.cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(self.data['x_test'])
        shap.summary_plot(shap_values,
                          self.data['x_test'],
                          plot_type="bar",
                          plot_size=(20, 16),
                          show=False)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(prop={'size': 18})
        plt.savefig(os.path.join(
            self.image_pth, 'results',
            "Random_Forest_Tree_Explainer.png"))

        # Calculate feature importances and sort in descending order
        importances = self.cv_rfc.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.feature_cols.columns[i] for i in indices]

        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance", fontsize=20)
        plt.ylabel('Importance', fontsize=18)
        plt.bar(range(self.feature_cols.shape[1]), importances[indices])
        plt.xticks(range(self.feature_cols.shape[1]),
                   names,
                   rotation=90,
                   fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(
            os.path.join(
                self.image_pth, 'results',
                "Random_Forest_Feature_Importances.png"))

        plt.close()

    def assess_models(self):
        """
        Compare performance of Random FOrest and Logistic Regression models.
        """

        # Train/Test predictions for Random Forest Model
        # KeyError occurs if training data does not exist;
        # AttributeError occurs if models do not exist
        # https://rollbar.com/blog/python-catching-multiple-exceptions/
        try:
            y_train_preds_rf = self.cv_rfc.best_estimator_.predict(
                self.data['x_train'])
            y_test_preds_rf = self.cv_rfc.best_estimator_.predict(
                self.data['x_test'])
        except (KeyError, AttributeError) as err:
            logging.error("""ERROR: Check that random forest model is
                          instantiated and trained, and that train/test data
                          exists.""")
            raise err

        # Train/Test predictions for Logistic Regression
        # KeyError occurs if training data does not exist;
        # AttributeError occurs if models do not exist
        try:
            y_train_preds_lr = self.lrc.predict(self.data['x_train'])
            y_test_preds_lr = self.lrc.predict(self.data['x_test'])
        except (KeyError, AttributeError) as err:
            logging.error("""ERROR: Check that logistic model is instantiated
                          and trained, and that train/test data exists.""")
            raise err

        # Check that result image paths exist and create if not
        self._check_path(os.path.join(self.image_pth, 'results'))

        # Generate Random Forest Classification Report
        self._generate_classification_report(
            y_train_preds_rf,
            y_test_preds_rf,
            "Random Forest")
        # Generate Logistic Regression Classification Report
        self._generate_classification_report(
            y_train_preds_lr,
            y_test_preds_lr,
            "Logistic Regression")

        # Plot ROC curves to compare performance of both models.
        self._plot_roc_curves()

        # Plot feature importances in Random Forest model
        self._assess_random_forest()

        logging.info("SUCCESS: Created model inference reports and results.")

    def _export_model(self, model: BaseEstimator, model_name: str):
        """
        Function to save trained models for predicting customer churn.

        Args:
            model (BaseEstimator): Trained sklearn model to save.
            model_name (str): Model name to use for file naming.
        """
        # If saving models, check model path exists
        self._check_path(self.model_pth)
        joblib.dump(model, os.path.join(self.model_pth, model_name + '.pkl'))

    def export_models(self):
        """
        Function to export trained models for future use.
        """

        # Check that export path exists and create if not
        self._check_path(self.model_pth)

        # Save Random Forest and Logistic Regression models
        self._export_model(self.cv_rfc, 'rfc_model')
        self._export_model(self.lrc, 'logistic_model')

        logging.info(
            "INFO: Saved Random Forest and Logistic Regression models in %s.",
            self.model_pth)

    def load_models(self, rfc_pth: str = "", lr_pth: str = ""):
        """
        Load pre-trained Random Forest and Logistic Regression models.

        Args:
            rfc_pth (str, optional): Path to pre-trained Random Forest model.
                Defaults to "".
            lr_pth (str, optional): Path to pre-trained Logistic Regression
                model. Defaults to "".
        """

        if rfc_pth:
            self.cv_rfc = joblib.load(rfc_pth)
            logging.info(
            "INFO: Imported Random Forest model from %s.", rfc_pth)

        if lr_pth:
            self.lrc = joblib.load(lr_pth)
            logging.info(
            "INFO: Imported Logistic Regression model from %s.", lr_pth)
