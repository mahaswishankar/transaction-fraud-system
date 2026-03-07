# ============================================================
# TRANSACTION FRAUD DETECTION - PART 3: MODEL TRAINING
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from xgboost import XGBClassifier
import joblib
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")
plt.rcParams['font.size'] = 12

# ── Paths ────────────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
output_dir    = os.path.join(script_dir, 'outputs')
processed_dir = os.path.join(script_dir, 'processed_data')
models_dir    = os.path.join(script_dir, 'models')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def save(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: outputs/{filename}")

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Processed Data")
print("=" * 60)

X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()

print(f"✅ X_train: {X_train.shape}  |  y_train fraud: {y_train.sum():,}")
print(f"✅ X_test : {X_test.shape}  |  y_test  fraud: {y_test.sum():,}")

# ============================================================
# 2. DEFINE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Defining Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

print("✅ 3 models defined:")
print("   1. Logistic Regression  — fast baseline, highly interpretable")
print("   2. Random Forest        — robust ensemble, handles non-linearity")
print("   3. XGBoost              — best performance, industry standard for fraud")

# ============================================================
# 3. TRAIN & EVALUATE ALL MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training & Evaluating All Models")
print("=" * 60)

results = {}

for name, model in models.items():
    print(f"\n  🔄 Training {name}...")
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    auc_roc     = roc_auc_score(y_test, y_prob)
    avg_prec    = average_precision_score(y_test, y_prob)
    report      = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'auc_roc'  : auc_roc,
        'avg_prec' : avg_prec,
        'precision': report['1']['precision'],
        'recall'   : report['1']['recall'],
        'f1'       : report['1']['f1-score'],
    }

    print(f"  ✅ {name} done!")
    print(f"     AUC-ROC   : {auc_roc:.4f}")
    print(f"     Avg Prec  : {avg_prec:.4f}")
    print(f"     Precision : {report['1']['precision']:.4f}")
    print(f"     Recall    : {report['1']['recall']:.4f}")
    print(f"     F1 Score  : {report['1']['f1-score']:.4f}")

    # Save model
    joblib.dump(model, os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl"))
    print(f"     💾 Model saved to models/")

# ============================================================
# 4. MODEL COMPARISON BAR CHART
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Model Comparison Chart")
print("=" * 60)

metrics    = ['auc_roc', 'precision', 'recall', 'f1']
labels     = ['AUC-ROC', 'Precision', 'Recall', 'F1 Score']
model_names = list(results.keys())
colors     = ['#3498db', '#2ecc71', '#e74c3c']

x     = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Model Comparison: All Metrics', fontsize=15, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
save('08_model_comparison.png')
plt.show()

# ============================================================
# 5. CONFUSION MATRICES
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Confusion Matrices")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f'{name}\nTP={tp} | FP={fp} | FN={fn} | TN={tn}',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold')
plt.tight_layout()
save('09_confusion_matrices.png')
plt.show()

print("\n  💡 Key interview point about confusion matrix:")
print("     FN (False Negatives) = fraud missed = COSTLY for JP Morgan")
print("     FP (False Positives) = legit tx blocked = bad customer experience")
print("     High Recall = fewer missed frauds (what banks prioritize)")

# ============================================================
# 6. ROC CURVES
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: ROC Curves")
print("=" * 60)

plt.figure(figsize=(10, 7))
colors = ['#3498db', '#2ecc71', '#e74c3c']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {res['auc_roc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curves — All Models', fontsize=15, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
save('10_roc_curves.png')
plt.show()

# ============================================================
# 7. PRECISION-RECALL CURVES
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Precision-Recall Curves")
print("=" * 60)
print("   (More informative than ROC for imbalanced datasets!)")

plt.figure(figsize=(10, 7))

for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    plt.plot(rec, prec, color=color, lw=2,
             label=f"{name} (AP = {res['avg_prec']:.4f})")

baseline = y_test.sum() / len(y_test)
plt.axhline(y=baseline, color='black', linestyle='--', lw=1.5,
            label=f'Random Classifier (AP = {baseline:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves — All Models', fontsize=15, fontweight='bold')
plt.legend(loc='upper right')
plt.tight_layout()
save('11_precision_recall_curves.png')
plt.show()

# ============================================================
# 8. FEATURE IMPORTANCE (XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Feature Importance (XGBoost)")
print("=" * 60)

xgb_model   = results['XGBoost']['model']
importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True).tail(20)

plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='#e74c3c', edgecolor='black', alpha=0.8)
plt.title('XGBoost — Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
save('12_feature_importance.png')
plt.show()

print("\n  Top 5 most important features:")
for feat, score in importances.tail(5).sort_values(ascending=False).items():
    print(f"     {feat:<20} : {score:.4f}")

# ============================================================
# 9. FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\n  {'Model':<25} {'AUC-ROC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Avg Prec':>10}")
print("  " + "-" * 75)
for name, res in results.items():
    print(f"  {name:<25} {res['auc_roc']:>8.4f} {res['precision']:>10.4f} {res['recall']:>8.4f} {res['f1']:>8.4f} {res['avg_prec']:>10.4f}")

best = max(results.items(), key=lambda x: x[1]['auc_roc'])
print(f"\n  🏆 Best model by AUC-ROC: {best[0]} ({best[1]['auc_roc']:.4f})")

# ============================================================
# 10. INTERVIEW TALKING POINTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: KEY POINTS (Talk about these at JP Morgan!)")
print("=" * 60)
print("""
  1. WHY NOT JUST USE ACCURACY?
     With 99.83% non-fraud, a model that predicts everything as
     non-fraud gets 99.83% accuracy but catches 0 frauds.
     -> Used AUC-ROC and Precision-Recall instead.

  2. WHY XGBoost WINS
     Handles non-linear patterns, robust to outliers,
     built-in regularization, and scale_pos_weight for imbalance.
     -> Industry standard for fraud detection at banks.

  3. BUSINESS TRADEOFF: PRECISION vs RECALL
     High Recall = catch more fraud (but more false alarms)
     High Precision = fewer false alarms (but miss some fraud)
     -> Banks typically prioritize Recall to minimize missed fraud.

  4. CONFUSION MATRIX INTERPRETATION
     False Negatives = fraud that slipped through = financial loss
     False Positives = blocked legit transactions = unhappy customers
     -> JP Morgan balances both using a threshold tuning strategy.

  5. MODELS SAVED TO disk
     All 3 models saved as .pkl files in models/ folder.
     -> Can be loaded and deployed without retraining.
""")

print("=" * 60)
print("✅ Part 3 Complete! Run Part 4 next: fraud_detection_part4_explainability.py")
print("=" * 60)
