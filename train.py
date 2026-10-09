import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load feature dataset
df = pd.read_csv("driver_pose_features.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# PCA
pca = PCA(
    n_components=20,
    random_state=42
)

X = pca.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy =", accuracy)

# Save everything
joblib.dump(
    model,
    "driver_model.pkl"
)

joblib.dump(
    encoder,
    "label_encoder.pkl"
)

joblib.dump(
    pca,
    "pca.pkl"
)

print("\nSaved:")
print("driver_model.pkl")
print("label_encoder.pkl")
print("pca.pkl")