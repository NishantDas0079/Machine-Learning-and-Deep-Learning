# simple_iris.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 60)
print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# 2. Create DataFrame for visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({i: name for i, name in enumerate(target_names)})

print("\nClass Distribution:")
print(df['species_name'].value_counts())

# 3. Simple visualization
plt.figure(figsize=(10, 6))
for i, species in enumerate(target_names):
    species_data = df[df['species'] == i]
    plt.scatter(species_data['sepal length (cm)'], 
                species_data['petal length (cm)'], 
                label=species, alpha=0.7, s=100)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Iris Dataset: Sepal Length vs Petal Length')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iris_scatter.png', dpi=150)
plt.show()

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Make predictions
y_pred = model.predict(X_test_scaled)

# 8. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 9. Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 10. Make a sample prediction
sample_flower = [[5.1, 3.5, 1.4, 0.2]]  # Should be Setosa
sample_scaled = scaler.transform(sample_flower)
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0]

print(f"\nSample Prediction:")
print(f"Features: {sample_flower[0]}")
print(f"Predicted species: {target_names[prediction]}")
print(f"Confidence: {max(probability):.2%}")

# 11. Save the model
import joblib
joblib.dump(model, 'iris_model.pkl')
print("\nModel saved as 'iris_model.pkl'")