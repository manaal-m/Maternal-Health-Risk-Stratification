import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv"
df = pd.read_csv(url)
df.columns = df.columns.str.strip()
print(f"Dataset Size: {df.shape}")

# Clean the dataset by removing outliers based on HeartRate (physiologically impossible values) 
df_clean = df[df['HeartRate'] > 10].copy() # Keeping only realistic heart rates
print(f"Dataset Size after Cleaning: {df_clean.shape}")

# Label Encoding 'RiskLevel'
le = LabelEncoder()
dict_map = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df_clean['RiskLevel_Encoded'] = df_clean['RiskLevel'].str.lower().map(dict_map)

# Check if it worked
print("\nClass Mapping Check:")
print(df_clean[['RiskLevel', 'RiskLevel_Encoded']].drop_duplicates())

# Save the cleaned dataset
df_clean.to_csv('maternal_health_cleaned.csv', index=False)
print(f"Data cleaning done. {len(df_clean)} records saved to maternal_health_cleaned.csv")