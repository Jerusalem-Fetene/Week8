# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clean import clean_df_credit, clean_df_fraud, clean_df_ip
import numpy as np


df_credit=pd.read_csv('../Data/credit_clean.csv')
df_fraud=pd.read_csv('../Data/fraud_clean.csv')
df_ip=pd.read_csv('../Data/ip_clean.csv')

# Clean the data frames
df_credit = clean_df_credit(df_credit)
df_fraud = clean_df_fraud(df_fraud)
df_ip = clean_df_ip(df_ip)

def univariate_analysis(df, column):
    print(f"Summary statistics for {column}:")
    print(df[column].describe())
    plt.figure(figsize=(10, 5))
    
    if np.issubdtype(df[column].dtype, np.number):
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
    else:
        plt.subplot(1, 2, 1)
        sns.countplot(x=df[column])
        plt.title(f'Count plot of {column}')
        
        plt.subplot(1, 2, 2)
        df[column].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie chart of {column}')
    
    plt.show()

def bivariate_analysis(df, col1, col2):
    print(f"Bivariate analysis between {col1} and {col2}:")
    plt.figure(figsize=(10, 5))
    
    if np.issubdtype(df[col1].dtype, np.number) and np.issubdtype(df[col2].dtype, np.number):
        sns.scatterplot(x=col1, y=col2, data=df)
        plt.title(f'Scatter plot between {col1} and {col2}')
    elif np.issubdtype(df[col1].dtype, np.number) and not np.issubdtype(df[col2].dtype, np.number):
        sns.boxplot(x=col2, y=col1, data=df)
        plt.title(f'Box plot of {col1} by {col2}')
    elif not np.issubdtype(df[col1].dtype, np.number) and np.issubdtype(df[col2].dtype, np.number):
        sns.boxplot(x=col1, y=col2, data=df)
        plt.title(f'Box plot of {col2} by {col1}')
    else:
        sns.countplot(x=col1, hue=col2, data=df)
        plt.title(f'Count plot of {col1} by {col2}')
    
    plt.show()


# if __name__ == "__main__":
#     # Univariate Analysis
#     univariate_analysis(df_credit, 'Amount')
#     univariate_analysis(df_credit, 'Class')

#     univariate_analysis(df_fraud, 'purchase_value')
#     univariate_analysis(df_fraud, 'browser')

#     univariate_analysis(df_ip, 'country')

#     # Bivariate Analysis
#     bivariate_analysis(df_credit, 'Amount', 'Class')
#     bivariate_analysis(df_credit, 'V1', 'V2')

#     bivariate_analysis(df_fraud, 'purchase_value', 'class')
#     bivariate_analysis(df_fraud, 'browser', 'class')

#     bivariate_analysis(df_ip, 'lower_bound_ip_address', 'country')
#     bivariate_analysis(df_ip, 'upper_bound_ip_address', 'country')
