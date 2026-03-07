# 💳 Transaction Fraud Detection
### A JP Morgan-ready Data Science Project

---

## 📁 Project Structure

```
fraud_detection_project/
│
├── data/
│   └── place creditcard.csv here        ← Download from Kaggle
│
├── outputs/
│   └── charts, models, reports saved here automatically
│
├── fraud_detection_part1_eda.py         ✅ Done
├── fraud_detection_part2_features.py    🔜 Coming soon
├── fraud_detection_part3_models.py      🔜 Coming soon
├── fraud_detection_part4_explainability.py  🔜 Coming soon
│
└── README.md
```

---

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
```

### Step 2: Download the dataset
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place `creditcard.csv` inside the `data/` folder.

### Step 3: Run parts in order
```bash
python fraud_detection_part1_eda.py
python fraud_detection_part2_features.py   # coming soon
python fraud_detection_part3_models.py     # coming soon
python fraud_detection_part4_explainability.py  # coming soon
```

---

## 📊 Project Parts

| Part | File | Status | What it does |
|------|------|--------|--------------|
| 1 | `fraud_detection_part1_eda.py` | ✅ Ready | Data exploration, class imbalance, patterns |
| 2 | `fraud_detection_part2_features.py` | 🔜 Soon | Feature engineering, scaling, SMOTE |
| 3 | `fraud_detection_part3_models.py` | 🔜 Soon | Logistic Regression, Random Forest, XGBoost |
| 4 | `fraud_detection_part4_explainability.py` | 🔜 Soon | SHAP values, model explainability |

---

## 🎯 Key Talking Points for JP Morgan Interview

1. **Class Imbalance** — Only 0.17% fraud. Used SMOTE to handle it.
2. **Right Metrics** — Used AUC-ROC, Precision, Recall — NOT just accuracy.
3. **Feature Engineering** — Created hour-of-day, amount bins from raw data.
4. **Model Explainability** — Used SHAP to explain WHY a transaction was flagged.
5. **Real-world thinking** — Considered false positive cost (annoying legit customers).

---

## 📦 Dataset Info
- **Source**: Kaggle - Credit Card Fraud Detection (ULB)
- **Size**: 284,807 transactions
- **Features**: 28 PCA features (V1–V28) + Amount + Time + Class
- **Fraud rate**: 0.172%
