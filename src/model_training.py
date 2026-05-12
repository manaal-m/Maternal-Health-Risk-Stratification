# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics & Validation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
df = pd.read_csv('maternal_health_cleaned.csv')
print(f"Original size: {len(df)}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Clean dataset size: {len(df)}")
le = LabelEncoder()
df['RiskLevel'] = le.fit_transform(df['RiskLevel'])
class_names = le.classes_
print(f"Class Mapping: {dict(zip(class_names, le.transform(class_names)))}")

# Define features and target
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
X = df[features]
y = df['RiskLevel']

# Split the dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features after splitting to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE is applied only to training data to prevent data leakage
print("\nApplying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameters are set for models 
param_grids = {
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30]
    },
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'probability': [True]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200, 500],
        'max_features': ['sqrt', 'log2']
    },
    "XGBoost": {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'use_label_encoder': [False],
        'eval_metric': ['mlogloss']
    }
}

final_models = {}
for name, params in param_grids.items():
    print(f"Tuning {name}...")
    if name == "Decision Tree": model = DecisionTreeClassifier(random_state=42)
    elif name == "SVM": model = SVC(random_state=42)
    elif name == "Random Forest": model = RandomForestClassifier(random_state=42)
    elif name == "XGBoost": model = XGBClassifier(random_state=42)
    
    grid = GridSearchCV(model, params, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    final_models[name] = grid.best_estimator_
    print(f"Best Params for {name}: {grid.best_params_}")


results = []
print("Training models... (Please wait)")

for name, model in final_models.items():
    # Train
    model.fit(X_train_bal, y_train_bal)

    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Recall': round(rec, 4),
        'Precision': round(prec, 4),
        'F1-Score': round(f1, 4)
    })

# Print final results 
results_df = pd.DataFrame(results)
print("\n----- FINAL RESULTS -----")
print(results_df)
