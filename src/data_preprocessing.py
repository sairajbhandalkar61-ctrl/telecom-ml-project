import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path):
    df = pd.read_csv(path, index_col='customerID')

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df