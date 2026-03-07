# ============================================================
# TRANSACTION FRAUD DETECTION - PART 1: EDA & DATA EXPLORATION
# Author: Your Name
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')

# ── Plot style ──────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ── Output directory (saves charts to outputs/ folder) ──────
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

def save(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: outputs/{filename}")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

# Auto-detect CSV location
for candidate in [
    os.path.join(script_dir, 'creditcard.csv'),
    os.path.join(script_dir, 'data', 'creditcard.csv'),
]:
    if os.path.exists(candidate):
        csv_path = candidate
        break
else:
    raise FileNotFoundError(
        "\n❌ Could not find creditcard.csv!\n"
        "   Place it in one of these locations:\n"
        f"   - {script_dir}\\creditcard.csv\n"
        f"   - {script_dir}\\data\\creditcard.csv\n"
        "   Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )

df = pd.read_csv(csv_path)
print(f"✅ Dataset loaded from: {csv_path}")
print(f"   Shape       : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")

# ============================================================
# 2. BASIC OVERVIEW
# ============================================================
print("=" * 60)
print("STEP 2: Basic Overview")
print("=" * 60)
print("\n📋 First 5 rows:")
print(df.head())
print("\n📊 Data types & null counts:")
print(df.info())
print("\n📈 Statistical summary:")
print(df.describe().T.to_string())

# ============================================================
# 3. CLASS DISTRIBUTION (Fraud vs Non-Fraud)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Class Distribution (Fraud vs Non-Fraud)")
print("=" * 60)

class_counts = df['Class'].value_counts()
class_pct    = df['Class'].value_counts(normalize=True) * 100

print(f"\n  Non-Fraud (0): {class_counts[0]:>7,}  ({class_pct[0]:.4f}%)")
print(f"  Fraud     (1): {class_counts[1]:>7,}  ({class_pct[1]:.4f}%)")
print(f"\n  Imbalance ratio: 1 fraud per every {class_counts[0]//class_counts[1]} non-fraud transactions")
print("  -> We MUST handle this imbalance before modelling (Part 3 covers SMOTE)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['Non-Fraud', 'Fraud'], class_counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(class_counts.values, labels=['Non-Fraud', 'Fraud'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.3f%%',
            startangle=90, explode=(0, 0.1))
axes[1].set_title('Class Distribution (Proportion)', fontsize=14, fontweight='bold')
plt.suptitle('Fraud vs Non-Fraud Transactions', fontsize=16, fontweight='bold')
plt.tight_layout()
save('01_class_distribution.png')
plt.show()

# ============================================================
# 4. TRANSACTION AMOUNT ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Transaction Amount Analysis")
print("=" * 60)

fraud     = df[df['Class'] == 1]['Amount']
non_fraud = df[df['Class'] == 0]['Amount']

print(f"\n  Fraud     -> Mean: ${fraud.mean():.2f}  |  Median: ${fraud.median():.2f}  |  Max: ${fraud.max():.2f}")
print(f"  Non-Fraud -> Mean: ${non_fraud.mean():.2f}  |  Median: ${non_fraud.median():.2f}  |  Max: ${non_fraud.max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(non_fraud, bins=80, alpha=0.6, color='#2ecc71', label='Non-Fraud', density=True)
axes[0].hist(fraud,     bins=80, alpha=0.7, color='#e74c3c', label='Fraud',     density=True)
axes[0].set_xlim(0, 500)
axes[0].set_title('Transaction Amount Distribution (<=\$500)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Amount ($)')
axes[0].set_ylabel('Density')
axes[0].legend()

axes[1].boxplot([non_fraud, fraud], labels=['Non-Fraud', 'Fraud'], patch_artist=True,
                boxprops=dict(facecolor='#2ecc71', alpha=0.6),
                medianprops=dict(color='black', linewidth=2))
axes[1].set_title('Amount Boxplot by Class', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Amount ($)')
axes[1].set_ylim(0, 500)
plt.suptitle('Transaction Amount: Fraud vs Non-Fraud', fontsize=15, fontweight='bold')
plt.tight_layout()
save('02_amount_analysis.png')
plt.show()

# ============================================================
# 5. TIME ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Transaction Time Analysis")
print("=" * 60)

df['Hour'] = (df['Time'] / 3600) % 24
fraud_hours     = df[df['Class'] == 1]['Hour']
non_fraud_hours = df[df['Class'] == 0]['Hour']

print(f"\n  Fraud peak hour    : {fraud_hours.value_counts().idxmax():.0f}:00")
print(f"  Non-Fraud peak hour: {non_fraud_hours.value_counts().idxmax():.0f}:00")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].hist(non_fraud_hours, bins=48, color='#2ecc71', alpha=0.7, label='Non-Fraud', density=True)
axes[0].set_title('Non-Fraud Transactions by Hour', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Density')
axes[0].legend()

axes[1].hist(fraud_hours, bins=48, color='#e74c3c', alpha=0.7, label='Fraud', density=True)
axes[1].set_title('Fraud Transactions by Hour', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.suptitle('Transaction Volume by Hour: Fraud vs Non-Fraud', fontsize=15, fontweight='bold')
plt.tight_layout()
save('03_time_analysis.png')
plt.show()

# ============================================================
# 6. FEATURE CORRELATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Feature Correlation with Fraud")
print("=" * 60)

correlations = df.corr()['Class'].drop('Class').sort_values()
print("\n  Top 5 POSITIVE correlations with fraud:")
print(correlations.tail(5).to_string())
print("\n  Top 5 NEGATIVE correlations with fraud:")
print(correlations.head(5).to_string())

plt.figure(figsize=(10, 8))
top_corr = pd.concat([correlations.head(10), correlations.tail(10)])
colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_corr.values]
top_corr.plot(kind='barh', color=colors, edgecolor='black', figsize=(10, 8))
plt.title('Top Feature Correlations with Fraud', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
save('04_feature_correlations.png')
plt.show()

# ============================================================
# 7. PCA FEATURE DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Top PCA Feature Distributions")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 5, figure=fig)
top_features = correlations.abs().sort_values(ascending=False).head(10).index.tolist()

for i, feature in enumerate(top_features):
    ax = fig.add_subplot(gs[i // 5, i % 5])
    ax.hist(df[df['Class'] == 0][feature], bins=50, alpha=0.6, color='#2ecc71', label='Non-Fraud', density=True)
    ax.hist(df[df['Class'] == 1][feature], bins=50, alpha=0.7, color='#e74c3c', label='Fraud',     density=True)
    ax.set_title(f'{feature}', fontweight='bold')
    ax.set_xlabel('Value')
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Top 10 Feature Distributions: Fraud vs Non-Fraud', fontsize=15, fontweight='bold')
plt.tight_layout()
save('05_feature_distributions.png')
plt.show()

# ============================================================
# 8. KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: KEY INSIGHTS (Talk about these in interviews!)")
print("=" * 60)
print("""
  1. SEVERE CLASS IMBALANCE
     Only 0.17% of transactions are fraud.
     -> Accuracy is MISLEADING here — must use Precision, Recall, AUC-ROC
     -> Must apply SMOTE / class_weight balancing (covered in Part 3)

  2. FRAUD AMOUNTS ARE SMALLER ON AVERAGE
     Fraudsters often test with small amounts first.
     -> Amount is useful but not the only signal

  3. TIME PATTERNS DIFFER
     Fraud has different hourly patterns vs normal transactions.
     -> Hour-of-day is a useful engineered feature

  4. PCA FEATURES (V1-V28) ARE MOST INFORMATIVE
     V17, V14, V12 show strong class separation.
     -> These will be top features in our model

  5. NO MISSING VALUES — clean dataset, no imputation needed.
     -> But we still scale Amount and Time (covered in Part 2)
""")

print("=" * 60)
print("✅ EDA Complete! Run Part 2 next: fraud_detection_part2_features.py")
print("=" * 60)
