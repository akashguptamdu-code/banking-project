import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("European_Bank.csv")

# 2. Feature engineering
def engineer(data):
    data = data.copy()
    data["BalanceToSalaryRatio"] = data["Balance"] / (data["EstimatedSalary"] + 1)
    data["ProductDensity"] = data["NumOfProducts"] / (data["Tenure"] + 1)
    data["EngagementProductInteraction"] = data["IsActiveMember"] * data["NumOfProducts"]
    data["AgeTenureInteraction"] = data["Age"] * data["Tenure"]
    return data

d = engineer(df)

# 3. Remove target and identifiers
drop_cols = ["Exited", "CustomerId", "Surname"]
if "Year" in d.columns:
    drop_cols.append("Year")
X = d.drop(columns=drop_cols)
y = d["Exited"]

categorical = ["Geography", "Gender"]
numeric = [c for c in X.columns if c not in categorical]

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# 5. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 6. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.90,
                              colsample_bytree=0.90, objective="binary:logistic",
                              eval_metric="logloss", random_state=42, n_jobs=-1)
}

results = []
pipelines = {}

# 7. Train and evaluate
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })
    pipelines[name] = pipe

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_results.csv", index=False)

# 8. Final XGBoost model
final_model = pipelines["XGBoost"]
joblib.dump(final_model, "bank_churn_xgboost_pipeline.joblib")

pred = final_model.predict(X_test)
prob = final_model.predict_proba(X_test)[:, 1]
print("\nXGBoost confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nXGBoost classification report:\n", classification_report(y_test, pred))

# 9. Score all customers
all_X = d.drop(columns=drop_cols)
scored = df.copy()
scored["ChurnProbability"] = final_model.predict_proba(all_X)[:, 1]
scored["PredictedChurn"] = (scored["ChurnProbability"] >= 0.50).astype(int)
scored["RiskBand"] = pd.cut(
    scored["ChurnProbability"],
    bins=[-0.01, 0.30, 0.60, 1.01],
    labels=["Low", "Medium", "High"]
)
scored.to_csv("bank_customer_churn_scored.csv", index=False)

print("\nSaved:")
print("model_comparison_results.csv")
print("bank_churn_xgboost_pipeline.joblib")
print("bank_customer_churn_scored.csv")
