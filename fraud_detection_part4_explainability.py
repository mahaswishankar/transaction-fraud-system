# ============================================================
# TRANSACTION FRAUD DETECTION - PART 4: EXPLAINABILITY (SHAP)
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")
plt.rcParams['font.size'] = 12

# ── Paths ────────────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
output_dir    = os.path.join(script_dir, 'outputs')
processed_dir = os.path.join(script_dir, 'processed_data')
models_dir    = os.path.join(script_dir, 'models')
os.makedirs(output_dir, exist_ok=True)

def save(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: outputs/{filename}")

# ============================================================
# 1. LOAD DATA & MODEL
# ============================================================
print("=" * 60)
print("STEP 1: Loading Data & XGBoost Model")
print("=" * 60)

X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()

model = joblib.load(os.path.join(models_dir, 'XGBoost.pkl'))

print(f"✅ Loaded XGBoost model from models/XGBoost.pkl")
print(f"✅ Test set: {X_test.shape[0]:,} rows | Fraud: {y_test.sum()} cases")

# ============================================================
# 2. SHAP EXPLAINER
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Computing SHAP Values")
print("=" * 60)
print("   SHAP = SHapley Additive exPlanations")
print("   Tells us WHY the model made each prediction")
print("   Based on game theory — each feature gets a 'contribution score'")
print("   Computing... (this takes ~1 min)\n")

# Use a sample for speed (500 fraud + 500 non-fraud)
fraud_idx     = y_test[y_test == 1].index[:500]
non_fraud_idx = y_test[y_test == 0].index[:500]
sample_idx    = fraud_idx.tolist() + non_fraud_idx.tolist()

X_sample = X_test.loc[sample_idx].reset_index(drop=True)
y_sample = y_test.loc[sample_idx].reset_index(drop=True)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print(f"✅ SHAP values computed for {len(X_sample)} transactions")
print(f"   Shape: {shap_values.shape}")

# ============================================================
# 3. SHAP SUMMARY PLOT (Global Feature Importance)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: SHAP Summary Plot — Global Feature Importance")
print("=" * 60)
print("   Shows which features matter MOST across all predictions")
print("   Red = high feature value, Blue = low feature value")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14, fontweight='bold')
plt.tight_layout()
save('13_shap_feature_importance.png')
plt.show()

# Dot plot (more informative)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary Plot — Feature Impact on Fraud Prediction", fontsize=13, fontweight='bold')
plt.tight_layout()
save('14_shap_summary_dot.png')
plt.show()

# ============================================================
# 4. SHAP FOR A REAL FRAUD TRANSACTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Explaining a Single FRAUD Transaction")
print("=" * 60)
print("   This is the most powerful thing to show in an interview!")
print("   'Why did the model flag THIS specific transaction as fraud?'")

# Pick the fraud transaction the model is most confident about
fraud_mask    = y_sample == 1
fraud_indices = X_sample[fraud_mask].index.tolist()
fraud_probs   = model.predict_proba(X_sample[fraud_mask])[:, 1]
most_confident_idx = fraud_indices[np.argmax(fraud_probs)]

transaction = X_sample.iloc[[most_confident_idx]]
prob        = model.predict_proba(transaction)[0][1]

print(f"\n   Selected transaction index : {most_confident_idx}")
print(f"   Model's fraud probability  : {prob:.4f} ({prob*100:.1f}% sure it's fraud)")

# SHAP waterfall-style bar chart for this transaction
shap_single = shap_values[most_confident_idx]
shap_df = pd.DataFrame({
    'Feature'   : X_sample.columns,
    'SHAP Value': shap_single,
    'Feature Val': transaction.values[0]
}).reindex(pd.Series(np.abs(shap_single)).sort_values(ascending=False).index)

top_shap = shap_df.head(15).sort_values('SHAP Value')

colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_shap['SHAP Value']]
plt.figure(figsize=(12, 7))
bars = plt.barh(
    [f"{r['Feature']} = {r['Feature Val']:.3f}" for _, r in top_shap.iterrows()],
    top_shap['SHAP Value'],
    color=colors, edgecolor='black', alpha=0.85
)
plt.axvline(x=0, color='black', linewidth=1)
plt.title(f"Why was this transaction flagged as FRAUD?\n(Model confidence: {prob*100:.1f}%)",
          fontsize=13, fontweight='bold')
plt.xlabel("SHAP Value (Red = pushes toward fraud, Blue = pushes toward legit)")
plt.tight_layout()
save('15_shap_single_fraud.png')
plt.show()

print("\n   Top 5 reasons this transaction was flagged:")
for _, row in shap_df.head(5).iterrows():
    direction = "→ FRAUD" if row['SHAP Value'] > 0 else "→ LEGIT"
    print(f"     {row['Feature']:<20}: SHAP={row['SHAP Value']:+.4f}  {direction}")

# ============================================================
# 5. SHAP FOR A LEGIT TRANSACTION (False Positive Analysis)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Explaining a Legitimate Transaction")
print("=" * 60)

legit_mask    = y_sample == 0
legit_indices = X_sample[legit_mask].index.tolist()
legit_probs   = model.predict_proba(X_sample[legit_mask])[:, 1]
legit_idx     = legit_indices[np.argmin(legit_probs)]

transaction_l = X_sample.iloc[[legit_idx]]
prob_l        = model.predict_proba(transaction_l)[0][1]

print(f"\n   Selected transaction index : {legit_idx}")
print(f"   Model's fraud probability  : {prob_l:.4f} ({prob_l*100:.2f}% — correctly identified as legit)")

shap_single_l = shap_values[legit_idx]
shap_df_l = pd.DataFrame({
    'Feature'    : X_sample.columns,
    'SHAP Value' : shap_single_l,
    'Feature Val': transaction_l.values[0]
}).reindex(pd.Series(np.abs(shap_single_l)).sort_values(ascending=False).index)

top_shap_l = shap_df_l.head(15).sort_values('SHAP Value')

colors_l = ['#e74c3c' if v > 0 else '#3498db' for v in top_shap_l['SHAP Value']]
plt.figure(figsize=(12, 7))
plt.barh(
    [f"{r['Feature']} = {r['Feature Val']:.3f}" for _, r in top_shap_l.iterrows()],
    top_shap_l['SHAP Value'],
    color=colors_l, edgecolor='black', alpha=0.85
)
plt.axvline(x=0, color='black', linewidth=1)
plt.title(f"Why was this transaction marked as LEGITIMATE?\n(Model confidence: {(1-prob_l)*100:.1f}% legit)",
          fontsize=13, fontweight='bold')
plt.xlabel("SHAP Value (Red = pushes toward fraud, Blue = pushes toward legit)")
plt.tight_layout()
save('16_shap_single_legit.png')
plt.show()

# ============================================================
# 6. SHAP FEATURE CORRELATION HEATMAP
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Top SHAP Features — Fraud vs Legit Comparison")
print("=" * 60)

top_features = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False).head(10).index.tolist()

fraud_shap = shap_values[y_sample == 1][:, [X_sample.columns.get_loc(f) for f in top_features]]
legit_shap = shap_values[y_sample == 0][:, [X_sample.columns.get_loc(f) for f in top_features]]

mean_fraud = fraud_shap.mean(axis=0)
mean_legit = legit_shap.mean(axis=0)

x      = np.arange(len(top_features))
width  = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, mean_fraud, width, label='Fraud Transactions',     color='#e74c3c', alpha=0.85, edgecolor='black')
ax.bar(x + width/2, mean_legit, width, label='Legit Transactions',     color='#2ecc71', alpha=0.85, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(top_features, rotation=45, ha='right')
ax.set_ylabel('Mean SHAP Value')
ax.set_title('Average SHAP Values: Fraud vs Legit Transactions', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.legend()
plt.tight_layout()
save('17_shap_fraud_vs_legit.png')
plt.show()

# ============================================================
# 7. FINAL PROJECT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE — FULL SUMMARY")
print("=" * 60)
print("""
  WHAT YOU BUILT:
  ──────────────
  A production-ready fraud detection system with:
  ✅ EDA on 284,807 real transactions
  ✅ Feature engineering (cyclic time, log amount, bins)
  ✅ SMOTE to handle 577:1 class imbalance
  ✅ 3 models trained: LR, Random Forest, XGBoost
  ✅ XGBoost: AUC-ROC ~0.98, F1 ~0.80
  ✅ SHAP explainability for every prediction

  FILES GENERATED:
  ────────────────
  outputs/  → 17 charts covering every aspect
  models/   → 3 trained .pkl model files
  processed_data/ → clean train/test CSVs

  HOW TO TALK ABOUT THIS AT JP MORGAN:
  ─────────────────────────────────────
  "I built a fraud detection system on 284k real transactions.
   The key challenge was a 577:1 class imbalance — I used SMOTE
   and chose AUC-ROC over accuracy as my metric. XGBoost achieved
   0.98 AUC-ROC and 0.80 F1. I also added SHAP explainability so
   analysts can understand WHY each transaction was flagged —
   which is critical for regulatory compliance at a bank."
""")
print("=" * 60)
print("✅ All 4 parts complete! Your project is JP Morgan ready. 🏦")
print("=" * 60)
