import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans, AgglomerativeClustering

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# 1. Load Data
data_path = "C:/Users/SH/Downloads/sprints/Heart_Disease_Project/heart+disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(data_path, names=column_names)

# 2. Handle Missing Values
df = df.replace("?", np.nan)
df["ca"] = pd.to_numeric(df["ca"], errors='coerce')
df["thal"] = pd.to_numeric(df["thal"], errors='coerce')
df = df.dropna()

# 3. Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# 4. Visualization - Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.savefig("visualizations/correlation_heatmap.png")
plt.close()

# 5. Scaling
X = df.drop("target", axis=1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualization - PCA Scatter Plot
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y.values
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', palette='Set2')
plt.title("PCA Projection")
plt.savefig("visualizations/pca_scatter.png")
plt.close()

# 7. Feature Selection
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=8)
X_selected = rfe.fit_transform(X_scaled, y)

# 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 9. Train Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

model_scores = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    model_scores[name] = auc
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("AUC:", auc)

# Visualization - AUC Comparison
plt.figure(figsize=(6, 4))
sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette='viridis')
plt.title("AUC Scores by Model")
plt.ylabel("AUC")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/auc_scores.png")
plt.close()

# 10. Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

agglo = AgglomerativeClustering(n_clusters=2)
agglo_labels = agglo.fit_predict(X_scaled)

# Visualization - KMeans Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set1')
plt.title("KMeans Clustering (on PCA)")
plt.savefig("visualizations/kmeans_clustering.png")
plt.close()

# 11. Hyperparameter Tuning
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest Logistic Regression Params:", grid.best_params_)

# 12. Save Final Model
joblib.dump(grid.best_estimator_, "models/final_model.pkl")
print("Model saved to models/final_model.pkl")

plt.show()
