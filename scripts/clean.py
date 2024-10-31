import pandas as pd
from sklearn.impute import SimpleImputer

df_credit = pd.read_csv('../Data/creditcard.csv')
df_fraud = pd.read_csv('../Data/Fraud_Data.csv')
df_ip = pd.read_csv('../Data/IpAddress_to_Country.csv')

# Function to clean df_credit
def clean_df_credit(df):
    # Handle Missing Values
    imputer = SimpleImputer(strategy='mean')
    df.iloc[:, :] = imputer.fit_transform(df)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Correct data types 
    df = df.astype({
        'Time': 'float64', 'V1': 'float64', 'V2': 'float64', 'V3': 'float64', 'V4': 'float64', 'V5': 'float64',
        'V6': 'float64', 'V7': 'float64', 'V8': 'float64', 'V9': 'float64', 'V10': 'float64', 'V11': 'float64',
        'V12': 'float64', 'V13': 'float64', 'V14': 'float64', 'V15': 'float64', 'V16': 'float64', 'V17': 'float64',
        'V18': 'float64', 'V19': 'float64', 'V20': 'float64', 'V21': 'float64', 'V22': 'float64', 'V23': 'float64',
        'V24': 'float64', 'V25': 'float64', 'V26': 'float64', 'V27': 'float64', 'V28': 'float64', 'Amount': 'float64',
        'Class': 'int64'
    })
    return df

# Function to clean df_fraud
def clean_df_fraud(df):
    # Convert time columns to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Handle Missing Values
    imputer = SimpleImputer(strategy='mean')
    df['ip_address'] = imputer.fit_transform(df[['ip_address']])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Correct data types
    df = df.astype({
        'user_id': 'int64', 'purchase_value': 'int64', 'device_id': 'object',
        'source': 'object', 'browser': 'object', 'sex': 'object', 'age': 'int64',
        'ip_address': 'float64', 'class': 'int64'
    })
    return df

# Function to clean df_ip
def clean_df_ip(df):
    # Handle Missing Values
    imputer = SimpleImputer(strategy='mean')
    df['lower_bound_ip_address'] = imputer.fit_transform(df[['lower_bound_ip_address']])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Correct data types
    df = df.astype({
        'lower_bound_ip_address': 'float64', 'upper_bound_ip_address': 'int64', 'country': 'object'
    })
    return df

# Clean the data frames
df_credit_cleaned = clean_df_credit(df_credit)
df_fraud_cleaned = clean_df_fraud(df_fraud)
df_ip_cleaned = clean_df_ip(df_ip)

# Save cleaned data frames 
df_credit_cleaned.to_csv('../Data/credit_clean.csv', index=False)
df_fraud_cleaned.to_csv('../Data/fraud_clean.csv', index=False)
df_ip_cleaned.to_csv('../Data/ip_clean.csv', index=False)

# Print to verify cleaning
print(df_credit_cleaned.info())
print(df_fraud_cleaned.info())
print(df_ip_cleaned.info())
