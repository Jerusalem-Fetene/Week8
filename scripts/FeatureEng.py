import pandas as pd

# Load the data
df_fraud = pd.read_csv('../Data/fraud_clean.csv')

# Convert 'purchase_time' to datetime format
df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])

# Extract hour of the day
df_fraud['hour_of_day'] = df_fraud['purchase_time'].dt.hour

# Extract day of the week
df_fraud['day_of_week'] = df_fraud['purchase_time'].dt.dayofweek

# Calculate transaction frequency per user per day
df_fraud['transaction_date'] = df_fraud['purchase_time'].dt.date
transaction_frequency = df_fraud.groupby(['user_id', 'transaction_date']).size().reset_index(name='transaction_count_per_day')

# Merge transaction frequency back to the original dataframe
df_fraud = df_fraud.merge(transaction_frequency, on=['user_id', 'transaction_date'], how='left')

# Calculate transaction velocity (rolling count of transactions in the past hour)
df_fraud = df_fraud.sort_values(by=['user_id', 'purchase_time'])

# Initialize a new column for transaction velocity
df_fraud['transaction_velocity_past_hour'] = 0

# Calculate the rolling count of transactions in the past hour for each user
for user_id in df_fraud['user_id'].unique():
    user_data = df_fraud[df_fraud['user_id'] == user_id].set_index('purchase_time').sort_index()
    rolling_count = user_data.rolling('1H').count() - 1
    df_fraud.loc[df_fraud['user_id'] == user_id, 'transaction_velocity_past_hour'] = rolling_count['user_id'].values

# Drop the intermediate 'transaction_date' column
df_fraud.drop(columns=['transaction_date'], inplace=True)

# Display the dataframe with new features
print(df_fraud.head())
