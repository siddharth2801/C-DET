import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def preprocess_data(file_path):
    # Load data
    data = pd.read_excel(file_path)

    # Handle target variable (Multi-Label Encoding for Diagnosis)
    data["Diagnosis_List"] = data["Diagnosis"].apply(eval)
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(data["Diagnosis_List"])

    # Normalize numerical features
    numerical_features = [
        'Frequency of Eating Out in a Week',
        'Frequency of Eating Out in a Month',
        'Total Meals in a Day',
        'How Much Veggies Eaten in a Day (gm)',
        'How Much Fruits Eaten in a Day (gm)'
    ]
    scaler = MinMaxScaler()
    numerical_scaled = scaler.fit_transform(data[numerical_features])

    # Encode categorical features using CountVectorizer
    categorical_features = data.select_dtypes(include=['object']).columns.drop("Diagnosis")
    vectorized_features = []
    for col in categorical_features:
        # Fill NaN values with a placeholder string
        data[col] = data[col].fillna("missing")
        
        # Flatten lists to strings if necessary
        data[col] = data[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Apply CountVectorizer
        vectorizer = CountVectorizer()
        feature_vector = vectorizer.fit_transform(data[col]).toarray()
        vectorized_features.append(feature_vector)

    # Combine all vectorized features
    categorical_encoded = np.hstack(vectorized_features)

    # Combine numerical and categorical features
    X = np.hstack([categorical_encoded, numerical_scaled])

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_binary, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, mlb
