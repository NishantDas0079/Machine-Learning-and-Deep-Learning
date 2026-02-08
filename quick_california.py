# Add these parameters to reduce time
QUICK_MODE = True  # Set to False for full run
N_ESTIMATORS = 50 if QUICK_MODE else 100
CV_FOLDS = 3 if QUICK_MODE else 5
TUNING_ITERATIONS = 10 if QUICK_MODE else 50

# quick_california.py - Faster version
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("Running quick version...")
housing = fetch_california_housing()

# Quick preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train only 2 models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    print(f"  {name} - R²: {results[name]['R2']:.4f}")

print("\n✅ Quick analysis complete in ~1 minute!")