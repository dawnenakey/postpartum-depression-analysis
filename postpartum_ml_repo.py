# postpartum_ml_repo
# Refactored and GitHub-Ready Machine Learning Code for Postpartum Depression Prediction

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define Paths
data_path = "data/final_merged_postpartum_data.csv"
output_path = "data/cleaned_postpartum_data.csv"

# Function: Load Data
def load_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"File {filepath} not found!")

# Function: Data Preprocessing
def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(columns=["Timestamp"], errors='ignore', inplace=True)
    
    # Fill missing values with mode
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_features = ["Feeling sad or Tearful", "Irritable towards baby & partner",
                            "Trouble sleeping at night", "Problems concentrating or making decision",
                            "Overeating or loss of appetite", "Feeling anxious", "Feeling of guilt",
                            "Problems of bonding with baby", "Suicide attempt"]
    
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure proper encoding
            label_encoders[col] = le  # Store encoders for future use
    
    # Encode Age groups
    age_mapping = {"18-25": 1, "25-30": 2, "30-35": 3, "35-40": 4, "40-45": 5, "45-50": 6}
    if "Age" in df.columns:
        df["Age"] = df["Age"].map(age_mapping)
    
    return df, label_encoders

# Function: Train Model
def train_model(df):
    X = df.drop(columns=["Suicide attempt"], errors='ignore')
    y = df["Suicide attempt"]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = xgb_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return xgb_model

# Run the pipeline
if __name__ == "__main__":
    try:
        df = load_data(data_path)
        df, encoders = preprocess_data(df)
        df.to_csv(output_path, index=False)  # Save cleaned data
        trained_model = train_model(df)
    except Exception as e:
        print("Error:", str(e))
