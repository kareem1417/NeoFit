import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import time

print("🚀 Starting PRODUCTION-READY ML Pipeline (No Data Leakage)...\n")

# 1. Load Data
df = pd.read_csv('fitness_dataset.csv')
X = df.drop('Recommended_Program_ID', axis=1)
y = df['Recommended_Program_ID']

# One-Hot Encoding (Done before split for simplicity in matching columns, doesn't leak target info)
categorical_cols = ['Sport_Type', 'Level', 'Goal']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
expected_features = list(X_encoded.columns)

# 2. The Golden Rule: Train/Test Split FIRST (80% Train, 20% Test)
# stratify=y ensures both sets have the same proportion of each program
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 3. Fit Preprocessing ONLY on Training Set to prevent leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform only, NO fitting!

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 4. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "KNN (K-Nearest)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("-" * 80)
print(f"{'Model Name':<25} | {'CV Accuracy (Train Set)':<25} | {'Status'}")
print("-" * 80)

best_model_name = ""
best_model_score = 0
best_model_obj = None

# 5. Cross-Validation on TRAINING SET ONLY
for name, model in models.items():
    # Using Macro F1 instead of simple accuracy to handle potential class imbalance
    cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5, scoring='f1_macro', n_jobs=-1)
    mean_score = np.mean(cv_scores)
    
    print(f"{name:<25} | {mean_score * 100:>12.2f}% (±{np.std(cv_scores)*100:.2f}%) | Tested")
    
    if mean_score > best_model_score:
        best_model_score = mean_score
        best_model_name = name
        best_model_obj = model

print("-" * 80)
print(f"\n🏆 Champion Selected for Tuning: ** {best_model_name} **\n")

# 6. Hyperparameter Tuning on TRAINING SET
print(f"⚙️ Tuning {best_model_name}...")
if best_model_name == "Random Forest":
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
                               param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_encoded)
    final_model = grid_search.best_estimator_

elif best_model_name == "Decision Tree":
    param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
                               param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_encoded)
    final_model = grid_search.best_estimator_

else:
    final_model = best_model_obj
    final_model.fit(X_train_scaled, y_train_encoded)

# 7. Final Evaluation on UNSEEN TEST SET (The Moment of Truth)
print("\n🧪 Evaluating Final Model on Unseen Test Data (20%)...")
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test_encoded, y_pred)

print(f"\n Final Unseen Test Accuracy: {test_accuracy * 100:.2f}%")
print("\n Detailed Classification Report:\n")
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0)
print(report)

# Save metrics to a file for presentation
metrics_summary = {
    "champion_model": best_model_name,
    "test_accuracy_percentage": round(test_accuracy * 100, 2),
    "evaluation_method": "80/20 Train/Test Split with 5-Fold CV on Train"
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

# 8. Save the robust pipeline
pipeline = {
    'model': final_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'features': expected_features
}
joblib.dump(pipeline, 'champion_model.pkl')

print("✅ Pipeline Saved! model_metrics.json generated for your presentation. 🚀")



# ==========================================
# 🎨 Data Visualization Section
# ==========================================
print("\n🎨 Generating Visualizations for Presentation...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(14, 10))
cm = confusion_matrix(y_test_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title(f"Confusion Matrix: {best_model_name} on Unseen Test Data", fontsize=16)
plt.xlabel("Predicted Program", fontsize=12)
plt.ylabel("Actual Program", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("   📸 Saved 'confusion_matrix.png'")

# 2. Feature Importance Bar Chart (If applicable)
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Get top 10 features
    top_n = 10
    top_features = [expected_features[i] for i in indices][:top_n]
    top_importances = importances[indices][:top_n]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title("Top 10 Features Driving the AI's Decisions", fontsize=16)
    plt.xlabel("Relative Importance (%)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    
    # Add percentage labels on the bars
    for index, value in enumerate(top_importances):
        plt.text(value, index, f" {value*100:.1f}%", va='center')
        
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("   📸 Saved 'feature_importance.png'")
# ==========================================
