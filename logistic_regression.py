import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

# 训练
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

# 评估
def evaluate(y_true, y_pred, y_prob, model_name):
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")

evaluate(y_test, y_pred, y_prob, "Logistic Regression")

results = {}

results["Logistic Regression"] = {
    "AUC-ROC": roc_auc_score(y_test, y_prob),
    "F1": f1_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred)
}

import json
with open("logistic_regression_results.json", "w") as f:
    json.dump(results, f, indent=2)