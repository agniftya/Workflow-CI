import pandas as pd
import mlflow
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.sklearn.autolog()
mlflow.set_experiment("Restaurant_Satisfaction_Experiment")

base_dir = os.path.dirname(os.path.abspath(__file__))

with mlflow.start_run():
    train_path = os.path.join(base_dir, 'train_preprocessed.csv')
    test_path = os.path.join(base_dir, 'test_preprocessed.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop('HighSatisfaction', axis=1)
    y_train = train_data['HighSatisfaction']
    X_test = test_data.drop('HighSatisfaction', axis=1)
    y_test = test_data['HighSatisfaction']

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Akurasi Model Dasar: {accuracy_score(y_test, y_pred):.4f}")

    mlflow.sklearn.log_model(model, "model")

joblib.dump(model, "model.pkl")