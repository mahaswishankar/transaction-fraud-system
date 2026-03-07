# ============================================================
# TRANSACTION FRAUD DETECTION - PART 2: FEATURE ENGINEERING
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")
plt.rcParams['font.size'] = 12

# ── Paths ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

def save(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: outputs/{filename}")

for candidate in [
    os.path.join(script_dir, 'creditcard.csv'),
    os.path.join(script_dir, 'data', 'creditcard.csv'),
]:
    if os.path.exists(candidate):
        csv_path = candidate
        break
else:
    raise FileNotFoundError("❌ creditcard.csv not found. See Part 1 instructions.")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(csv_path)
print(f"✅ Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# 2a. Hour of day (cyclic) — captures time patterns from EDA
df['Hour']     = (df['Time'] / 3600) % 24
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
print("✅ Created Hour, Hour_sin, Hour_cos (cyclic time encoding)")

# 2b. Amount bins — fraudsters often use small test amounts
df['Amount_bin'] = pd.cut(
    df['Amount'],
    bins=[-0.001, 1, 10, 50, 200, 500, df['Amount'].max() + 1],
    labels=[0, 1, 2, 3, 4, 5]
).cat.add_categories([-1]).fillna(-1).astype(int)
print("✅ Created Amount_bin (0=micro, 1=tiny, 2=small, 3=medium, 4=large, 5=huge)")

# 2c. Log transform of Amount (reduces skewness)
df['Amount_log'] = np.log1p(df['Amount'])
print("✅ Created Amount_log (log-transformed amount)")

# 2d. Is night transaction (fraud peaks at night from EDA)
df['Is_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
print("✅ Created Is_night (1 = transaction between 10pm–6am)")

print(f"\n   Total features now: {df.shape[1]}")

# ============================================================
# 3. SCALING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Scaling (RobustScaler)")
print("=" * 60)

# RobustScaler is better than StandardScaler for fraud data
# because it's resistant to outliers (big transaction amounts)
scaler = RobustScaler()

df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler.fit_transform(df[['Time']])
print("✅ Scaled Amount and Time using RobustScaler")
print("   (RobustScaler chosen because fraud data has many outliers)")

# Visualize before vs after scaling
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].hist(df['Amount'], bins=100, color='#e74c3c', alpha=0.7)
axes[0, 0].set_title('Amount - BEFORE Scaling', fontweight='bold')
axes[0, 0].set_xlabel('Amount')
axes[0, 0].set_xlim(0, 1000)

axes[0, 1].hist(df['Amount_scaled'], bins=100, color='#2ecc71', alpha=0.7)
axes[0, 1].set_title('Amount - AFTER Scaling', fontweight='bold')
axes[0, 1].set_xlabel('Scaled Amount')

axes[1, 0].hist(df['Time'], bins=100, color='#e74c3c', alpha=0.7)
axes[1, 0].set_title('Time - BEFORE Scaling', fontweight='bold')
axes[1, 0].set_xlabel('Time (seconds)')

axes[1, 1].hist(df['Time_scaled'], bins=100, color='#2ecc71', alpha=0.7)
axes[1, 1].set_title('Time - AFTER Scaling', fontweight='bold')
axes[1, 1].set_xlabel('Scaled Time')

plt.suptitle('Feature Scaling: Before vs After', fontsize=15, fontweight='bold')
plt.tight_layout()
save('06_scaling.png')
plt.show()

# ============================================================
# 4. PREPARE FEATURES & TARGET
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Preparing Feature Matrix (X) and Target (y)")
print("=" * 60)

# Drop original unscaled columns and keep engineered ones
drop_cols = ['Time', 'Amount', 'Hour', 'Class']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df['Class']

print(f"✅ Feature matrix X: {X.shape}")
print(f"✅ Target vector  y: {y.shape}")
print(f"\n   Features used ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2}. {col}")

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Train / Test Split (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves fraud ratio in both splits
)

print(f"✅ Train set : {X_train.shape[0]:,} rows  | Fraud: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"✅ Test set  : {X_test.shape[0]:,} rows  | Fraud: {y_test.sum():,}  ({y_test.mean()*100:.3f}%)")
print("   (stratify=y ensures fraud ratio is preserved in both splits)")

# ============================================================
# 6. SMOTE — Fix Class Imbalance
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Applying SMOTE to Training Data")
print("=" * 60)
print("   SMOTE = Synthetic Minority Over-sampling Technique")
print("   Creates synthetic fraud samples so model can learn better")
print("   ⚠️  SMOTE applied ONLY to training data (never test data!)\n")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"   BEFORE SMOTE -> Fraud: {y_train.sum():,}  |  Non-Fraud: {(y_train==0).sum():,}")
print(f"   AFTER  SMOTE -> Fraud: {y_train_sm.sum():,}  |  Non-Fraud: {(y_train_sm==0).sum():,}")
print(f"\n✅ Training set is now perfectly balanced!")

# Visualize SMOTE effect
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

before = [y_train.sum(), (y_train == 0).sum()]
after  = [y_train_sm.sum(), (y_train_sm == 0).sum()]

axes[0].bar(['Fraud', 'Non-Fraud'], before, color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title('BEFORE SMOTE', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(before):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

axes[1].bar(['Fraud', 'Non-Fraud'], after, color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[1].set_title('AFTER SMOTE', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(after):
    axes[1].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

plt.suptitle('Class Balance: Before vs After SMOTE', fontsize=15, fontweight='bold')
plt.tight_layout()
save('07_smote_balance.png')
plt.show()

# ============================================================
# 7. SAVE PROCESSED DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Processed Data")
print("=" * 60)

processed_dir = os.path.join(script_dir, 'processed_data')
os.makedirs(processed_dir, exist_ok=True)

X_train_sm.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'),  index=False)
pd.Series(y_train_sm, name='Class').to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
pd.Series(y_test,     name='Class').to_csv(os.path.join(processed_dir, 'y_test.csv'),  index=False)

print(f"✅ Saved to processed_data/ folder:")
print(f"   - X_train.csv ({X_train_sm.shape[0]:,} rows, SMOTE applied)")
print(f"   - X_test.csv  ({X_test.shape[0]:,} rows, original)")
print(f"   - y_train.csv")
print(f"   - y_test.csv")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: KEY POINTS (Talk about these in interviews!)")
print("=" * 60)
print("""
  1. FEATURE ENGINEERING
     Created Hour_sin/cos (cyclic), Amount_log, Amount_bin, Is_night
     -> These capture domain-specific fraud patterns

  2. ROBUST SCALER (not StandardScaler)
     RobustScaler uses median/IQR instead of mean/std
     -> Not affected by extreme transaction outliers

  3. STRATIFIED SPLIT
     Preserves the 0.17% fraud ratio in both train and test sets
     -> Ensures test set reflects real-world distribution

  4. SMOTE ON TRAINING DATA ONLY
     Never apply SMOTE to test data — that would leak information
     -> Test data must always reflect real-world distribution

  5. WHY NOT JUST UNDERSAMPLE?
     Undersampling loses 99% of non-fraud data
     SMOTE creates new synthetic samples instead — much better
""")

print("=" * 60)
print("✅ Part 2 Complete! Run Part 3 next: fraud_detection_part3_models.py")
print("=" * 60)
