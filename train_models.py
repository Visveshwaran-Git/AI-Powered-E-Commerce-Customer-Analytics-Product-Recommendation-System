import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from data_generator import generate_customer_data

# Ensure data and models directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

def preprocess_data(filepath='data/customer_data.csv'):
    # Generate data if missing
    if not os.path.exists(filepath):
        generate_customer_data(filepath)
        
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # 1. Handle missing values
    # Fill numerical with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    # 2. Remove duplicates
    df = df.drop_duplicates()
    
    # 3. Encode categorical features
    label_encoder = LabelEncoder()
    df['FavoriteCategory_Encoded'] = label_encoder.fit_transform(df['FavoriteCategory'])
    
    # Save the encoder for later use
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    # Features and Target
    features = ['Age', 'Income', 'FavoriteCategory_Encoded', 'BrowsingTimeMins', 'PagesVisited', 'PastPurchases']
    X = df[features]
    y = df['Purchased']
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save processed test data for explainability/visualization later if needed
    pd.DataFrame(X_test_scaled, columns=features).to_csv('data/X_test_scaled.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/logistic_regression.pkl')
    return acc

def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest Classifier...")
    # Basic hyperparameter tuning (using predefined optimal params for speed)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {acc:.4f}")
    
    # Save model
    joblib.dump(model, 'models/random_forest.pkl')
    return acc

def train_tensorflow_nn(X_train, y_train, X_test, y_test):
    print("Training TensorFlow Neural Network...")
    
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow NN Accuracy: {acc:.4f}")
    
    # Save model
    model.save('models/tensorflow_model.h5')
    
    # Save training history for visualization
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('data/tf_history.csv', index=False)
    
    return acc

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = preprocess_data()
    
    lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    tf_acc = train_tensorflow_nn(X_train, y_train, X_test, y_test)
    
    # Save accuracies for visualization
    acc_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Neural Network'],
        'Accuracy': [lr_acc, rf_acc, tf_acc]
    })
    acc_df.to_csv('data/model_accuracies.csv', index=False)
    print("Model training complete.")
