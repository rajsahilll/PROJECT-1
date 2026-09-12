"""
03_logistic_regression_model.py
E-commerce Return Rate Reduction Analysis — Step 3: Predictive Modeling

Trains a logistic regression model to predict P(Return) per order using
only features known AT ORDER TIME (no target leakage from Return_Reason,
Return_Date, Days_to_Return, which only exist for already-returned orders).

Outputs:
  outputs/model_metrics.txt
  outputs/feature_importance.csv
  outputs/powerbi_dataset.csv          (full dataset + Risk_Score + Risk_Tier)
  outputs/high_risk_products.csv       (product-level rollup, sorted by risk)
  charts/confusion_matrix.png
  charts/roc_curve.png
  charts/feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)

RANDOM_STATE = 42

df = pd.read_csv("outputs/cleaned_ecommerce_returns.csv")

# -------------------------------------------------------------------
# 1. Feature set — order-time information only (no leakage)
# -------------------------------------------------------------------
numeric_features = ["Product_Price", "Order_Quantity", "User_Age", "Discount_Applied"]
categorical_features = ["Product_Category", "User_Gender", "Payment_Method", "Shipping_Method"]

# User_Location has 100 levels -> use smoothed target-mean encoding (fit on train only)
target = "Is_Returned"

X = df[numeric_features + categorical_features + ["User_Location"]].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# -- Smoothed mean-target encoding for User_Location (fit on train only, avoids leakage) --
global_mean = y_train.mean()
smoothing = 20
loc_stats = X_train.assign(y=y_train.values).groupby("User_Location")["y"].agg(["mean", "count"])
loc_stats["smoothed"] = (loc_stats["mean"] * loc_stats["count"] + global_mean * smoothing) / (loc_stats["count"] + smoothing)
loc_map = loc_stats["smoothed"].to_dict()

X_train["Location_Risk_Encoded"] = X_train["User_Location"].map(loc_map).fillna(global_mean)
X_test["Location_Risk_Encoded"] = X_test["User_Location"].map(loc_map).fillna(global_mean)

# also encode full dataset for the Power BI export later
df["Location_Risk_Encoded"] = df["User_Location"].map(loc_map).fillna(global_mean)

numeric_features_final = numeric_features + ["Location_Risk_Encoded"]

X_train_model = X_train.drop(columns=["User_Location"])
X_test_model = X_test.drop(columns=["User_Location"])

# -------------------------------------------------------------------
# 2. Preprocessing + Logistic Regression pipeline
# -------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features_final),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

model.fit(X_train_model, y_train)

# -------------------------------------------------------------------
# 3. Evaluation
# -------------------------------------------------------------------
y_pred = model.predict(X_test_model)
y_proba = model.predict_proba(X_test_model)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

metrics_text = f"""LOGISTIC REGRESSION — RETURN PREDICTION MODEL
================================================
Train rows: {len(X_train_model)}   Test rows: {len(X_test_model)}
Baseline (majority class) accuracy: {max(y_test.mean(), 1-y_test.mean()):.4f}

Accuracy : {acc:.4f}
Precision: {prec:.4f}
Recall   : {rec:.4f}
F1-score : {f1:.4f}
ROC-AUC  : {auc:.4f}

Confusion Matrix (rows=actual, cols=predicted) [0=Not Returned, 1=Returned]:
{cm}

Classification Report:
{report}

NOTE ON MODEL PERFORMANCE:
This dataset's return outcome is generated close to independent of the
available features (return rate hovers near 50% in nearly every slice —
see EDA outputs). As a result, ROC-AUC is close to 0.50, meaning these
order-level attributes alone carry very little real predictive signal
in this particular dataset. The pipeline, encoding strategy, evaluation
and risk-scoring approach below are fully production-ready and will
scale up in predictive power automatically if applied to a real dataset
where returns are actually driven by these features (e.g. real defect
rates by supplier, real fit issues by category/size, etc).
"""

with open("outputs/model_metrics.txt", "w") as f:
    f.write(metrics_text)
print(metrics_text)

# Confusion matrix chart
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Returned", "Returned"], yticklabels=["Not Returned", "Returned"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random guess (AUC=0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Return Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("charts/roc_curve.png", dpi=150)
plt.close()

# -------------------------------------------------------------------
# 4. Feature importance (standardized logistic regression coefficients)
# -------------------------------------------------------------------
feature_names = (numeric_features_final +
                  list(model.named_steps["preprocess"]
                       .named_transformers_["cat"]
                       .get_feature_names_out(categorical_features)))
coefs = model.named_steps["clf"].coef_[0]
fi = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
fi["Abs_Coefficient"] = fi["Coefficient"].abs()
fi = fi.sort_values("Abs_Coefficient", ascending=False)
fi.to_csv("outputs/feature_importance.csv", index=False)

plt.figure(figsize=(8, 6))
top_fi = fi.head(15)
colors = ["#C44E52" if c > 0 else "#4C72B0" for c in top_fi["Coefficient"]]
plt.barh(top_fi["Feature"][::-1], top_fi["Coefficient"][::-1], color=colors[::-1])
plt.xlabel("Coefficient (impact on log-odds of return)")
plt.title("Top Feature Importance — Logistic Regression")
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=150)
plt.close()

print("Top features:\n", fi.head(10), "\n")

# -------------------------------------------------------------------
# 5. Score ALL orders (full dataset) -> Risk_Score + Risk_Tier
# -------------------------------------------------------------------
X_full = df[numeric_features_final + categorical_features].copy()
df["Risk_Score"] = model.predict_proba(X_full)[:, 1].round(4)

df["Risk_Tier"] = pd.cut(df["Risk_Score"], bins=[-0.01, 0.40, 0.60, 1.0],
                          labels=["Low", "Medium", "High"])

df.to_csv("outputs/powerbi_dataset.csv", index=False)
print(f"Power BI-ready dataset saved: outputs/powerbi_dataset.csv ({len(df)} rows)")

# -------------------------------------------------------------------
# 6. Product-level high-risk rollup (deliverable: CSV of high-risk products)
# -------------------------------------------------------------------
product_risk = df.groupby(["Product_ID", "Product_Category"]).agg(
    Orders=("Order_ID", "count"),
    Actual_Returns=("Is_Returned", "sum"),
    Actual_Return_Rate=("Is_Returned", "mean"),
    Avg_Risk_Score=("Risk_Score", "mean"),
    Avg_Price=("Product_Price", "mean"),
    Avg_Discount=("Discount_Applied", "mean"),
).reset_index()
product_risk["Actual_Return_Rate"] = (product_risk["Actual_Return_Rate"] * 100).round(2)
product_risk["Avg_Risk_Score"] = product_risk["Avg_Risk_Score"].round(4)
product_risk["Avg_Price"] = product_risk["Avg_Price"].round(2)
product_risk["Avg_Discount"] = product_risk["Avg_Discount"].round(2)

product_risk = product_risk.sort_values("Avg_Risk_Score", ascending=False)
product_risk["Risk_Rank"] = range(1, len(product_risk) + 1)

high_risk = product_risk[product_risk["Avg_Risk_Score"] >= product_risk["Avg_Risk_Score"].quantile(0.75)]
high_risk.to_csv("outputs/high_risk_products.csv", index=False)
print(f"High-risk products exported: outputs/high_risk_products.csv ({len(high_risk)} products, "
      f"top quartile by predicted risk)")
