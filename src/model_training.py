# Libtaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report
)

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Model selection
from sklearn.model_selection import train_test_split, GridSearchCV

# Pipeline + SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


#Load data
df = pd.read_csv('maternal_health_cleaned.csv')

print(f"Original size: {len(df)}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Clean dataset size: {len(df)}")

# Encode target
le = LabelEncoder()
df['RiskLevel'] = le.fit_transform(df['RiskLevel'])
class_names = le.classes_

print("Class Mapping:", dict(zip(class_names, le.transform(class_names))))


#Features & Target
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
X = df[features]
y = df['RiskLevel']


#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#Pipelines for each model
pipelines = {

    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', SVC(probability=True, random_state=42))
    ]),

    "Decision Tree": Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42))
    ]),

    "XGBoost": Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(random_state=42, eval_metric='mlogloss'))
    ])
}


#Hyperparameter grids for each model
param_grids = {

    "SVM": {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },

    "Decision Tree": {
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': [None, 10, 20]
    },

    "Random Forest": {
        'model__n_estimators': [100, 200],
        'model__max_features': ['sqrt', 'log2']
    },

    "XGBoost": {
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5, 7]
    }
}


#Train models with GridSearchCV
final_models = {}

for name in pipelines:
    print(f"\nTuning {name}...")

    grid = GridSearchCV(
        pipelines[name],
        param_grids[name],
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    final_models[name] = grid.best_estimator_
    print("Best Params:", grid.best_params_)


#Evaluate models on test set
results = []

for name, model in final_models.items():

    y_pred = model.predict(X_test)

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

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))


results_df = pd.DataFrame(results)
print("\n===== FINAL RESULTS =====")
print(results_df)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(final_models["XGBoost"], X, y, cv=5, scoring='f1_macro')
print("CV F1 scores:", scores)
print("Mean F1:", scores.mean())