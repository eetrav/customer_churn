"""
Collection of plotters for exploratory data analysis on Pandas dataframe.

Author: Emily Travinsky
Date: 09/2023
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.size'] = 22


class Plotters():
    """
    Class to track and generate plots of Pandas dataframe.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame = pd.DataFrame(),
            image_output: str = "./images"):
        """
        Instantiation of relevant variables for plotting.

        Originally intended for bank data to track customer churn, but can be
        used for any Pandas dataframe.

        Args:
            dataframe (pd.DataFrame, optional): Pandas dataframe.
                Defaults to None.
            image_output (str, optional): Path to image output folder. Defaults
                to "./images".
        """
        self.data = dataframe
        self.image_pth = image_output

    def plot_hist(self, col: str = ''):
        """
        Function to plot histogram of customer data.

        Args:
            bank_df (pd.DataFrame, optional): [description]. Defaults to None.
            col (str, optional): [description]. Defaults to ''.
        """

        plt.figure(figsize=(20, 10))
        plt.rcParams.update({'font.size': 22})
        self.data[col].hist()
        plt.xlabel(col, fontsize=18)
        plt.ylabel('Number of Occurrences', fontsize=18)
        plt.title(col + " Histogram", fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(os.path.join(self.image_pth, col + '_hist.png'))

    def plot_counts(self, col: str = ''):
        """
        Function to plot value counts of customer data.

        Args:
            col (str, optional): Name of dataframe column to plot value counts.
                Defaults to ''.
        """

        plt.figure(figsize=(20, 10))
        self.data[col].value_counts(
            'normalize').plot(kind='bar')
        plt.xlabel(col, fontsize=18)
        plt.ylabel('Counts', fontsize=18)
        plt.title(col + " Value Counts", fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(os.path.join(self.image_pth, col + '_counts.png'))

    def plot_dist(self, col: str = ''):
        """
        Function to plot distribution of customer data.

        Args:
            col (str, optional): Name of dataframe column to plot distribution.
                Defaults to ''.
        """

        plt.figure(figsize=(20, 10))
        # distplot is deprecated. Use histplot instead
        # sns.distplot(df['Total_Trans_Ct']);
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
        # using a kernel density estimate
        sns.histplot(
            self.data[col],
            stat='density',
            kde=True)
        plt.xlabel(col, fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.title(col + " Distribution", fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(os.path.join(self.image_pth, col + '_dist.png'))

    def plot_corr(self):
        """
        Function to plot correlations of dataframe columns.
        """

        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self.data.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.title("Data Correlation", fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(os.path.join(self.image_pth, 'Correlation.png'))
