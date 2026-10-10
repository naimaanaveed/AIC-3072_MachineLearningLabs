
# Activity 1: KNN for Credit Card Fraud Detection

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Step 2: Load the Dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df["Class"].value_counts())

# Step 3: Split Features (X) and Target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 4: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Step 5: Create a Pipeline
# Scaling and SMOTE are applied only to training folds.
# SMOTE sampling_strategy=0.1 reduces computational cost.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(
        sampling_strategy=0.1,
        random_state=42
    )),
    ("knn", KNeighborsClassifier())
])

# Step 6: Train KNN Classifier with k=5
pipeline.set_params(knn__n_neighbors=5)
pipeline.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = pipeline.predict(X_test)

# Step 8: Evaluate the Initial Model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- KNN Results (k=5) ---")
print("Accuracy:", accuracy)
print("F1-Score:", f1)

# Step 9: Tune k Using GridSearchCV
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1-Score:", grid_search.best_score_)

# Step 10: Evaluate the Best Model on Test Data
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

print("\n--- Best KNN Model Results ---")
print("Best Accuracy:", best_accuracy)
print("Best F1-Score:", best_f1)

# Step 11: Display Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))
