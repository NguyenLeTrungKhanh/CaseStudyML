# ============================================================
# CASE STUDY: Network Intrusion Detection using Deep Learning
# Dataset  : UNSW-NB15
# Models   : Sequential DNN (8 features) + 1D-CNN (43 features)
# ============================================================

# Cài đặt thư viện:
# pip install pandas numpy scikit-learn tensorflow matplotlib seaborn imbalanced-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score,
                              precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                      Flatten, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Seed cho tái hiện kết quả
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# BƯỚC 1: TẢI DỮ LIỆU UNSW-NB15
# ============================================================
# Tải dataset tại: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
# Đặt file CSV cùng thư mục và đặt tên: UNSW_NB15_training-set.csv

print("=" * 60)
print("BUOC 1: TAI VA KHAM PHA DU LIEU")
print("=" * 60)

try:
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    print(f"Doc file thanh cong! Shape: {df.shape}")
except FileNotFoundError:
    print("Khong tim thay file CSV. Tao du lieu mau de demo...")
    np.random.seed(42)
    n = 8000
    feature_names = [
        'id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
        'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
        'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
        'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
        'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
        'ct_srv_dst', 'is_sm_ips_ports'
    ]
    df = pd.DataFrame(np.abs(np.random.randn(n, len(feature_names))),
                      columns=feature_names)
    df['proto']   = np.random.choice(['tcp', 'udp', 'icmp'], n)
    df['service'] = np.random.choice(['-', 'http', 'ftp', 'smtp'], n)
    df['state']   = np.random.choice(['FIN', 'INT', 'CON', 'REQ'], n)
    df['label']   = np.random.choice([0, 1], n, p=[0.35, 0.65])
    print(f"Tao du lieu mau thanh cong! Shape: {df.shape}")

print(f"\nPhan phoi nhan:\n{df['label'].value_counts()}")
print(f"Kieu du lieu:\n{df.dtypes.value_counts()}")

# ============================================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU
# ============================================================
print("\n" + "=" * 60)
print("BUOC 2: TIEN XU LY DU LIEU")
print("=" * 60)

drop_cols = ['label']
if 'attack_cat' in df.columns:
    drop_cols.append('attack_cat')

X = df.drop(columns=drop_cols).copy()
y = df['label'].values

# Label Encoding cho biến categorical
cat_cols = X.select_dtypes(include=['object']).columns
print(f"Cot categorical: {list(cat_cols)}")
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Xử lý missing values
X = X.fillna(X.median())

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Shape sau tien xu ly: {X_scaled.shape}")

# SMOTE: cân bằng dữ liệu
print("Ap dung SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"Shape sau SMOTE: {X_resampled.shape}")
print(f"Phan phoi nhan sau SMOTE: {np.bincount(y_resampled)}")

# ============================================================
# BƯỚC 3: CHỌN ĐẶC TRƯNG VỚI EXTRA TREE CLASSIFIER
# ============================================================
print("\n" + "=" * 60)
print("BUOC 3: CHON DAC TRUNG - EXTRA TREE CLASSIFIER")
print("=" * 60)

etc = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc.fit(X_resampled, y_resampled)

importances = pd.Series(etc.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

THRESHOLD = 0.021  # Ngưỡng theo bài báo
selected_features = importances[importances >= THRESHOLD].index.tolist()

print(f"\nNguong chon dac trung: {THRESHOLD}")
print(f"So dac trung duoc chon: {len(selected_features)}")
print(f"\n{'Feature':<25} {'Importance':>12}")
print("-" * 38)
for f in selected_features:
    print(f"{f:<25} {importances[f]:>12.4f}")

# Biểu đồ Feature Importance
plt.figure(figsize=(12, 6))
top15 = importances_sorted.head(15)
colors = ['#2196F3' if imp >= THRESHOLD else '#90CAF9' for imp in top15.values]
bars = plt.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
plt.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2,
            label=f'Threshold = {THRESHOLD}')
plt.title('Feature Importance - Extra Tree Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.legend()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nLuu: feature_importance.png")

# Tách features đã chọn
selected_idx = [list(X.columns).index(f) for f in selected_features
                if f in X.columns]
X_selected = X_resampled[:, selected_idx]

# ============================================================
# BƯỚC 4: CHIA DỮ LIỆU TRAIN / TEST
# ============================================================
print("\n" + "=" * 60)
print("BUOC 4: CHIA DU LIEU TRAIN/TEST (70/30)")
print("=" * 60)

# DNN: dùng 8 features đã chọn
X_tr_dnn, X_te_dnn, y_tr, y_te = train_test_split(
    X_selected, y_resampled, test_size=0.30, random_state=42, stratify=y_resampled)

# CNN: dùng tất cả 43 features
X_tr_cnn, X_te_cnn, y_tr_cnn, y_te_cnn = train_test_split(
    X_resampled, y_resampled, test_size=0.30, random_state=42, stratify=y_resampled)

# One-hot encoding
y_tr_cat      = to_categorical(y_tr,      num_classes=2)
y_te_cat      = to_categorical(y_te,      num_classes=2)
y_tr_cnn_cat  = to_categorical(y_tr_cnn,  num_classes=2)
y_te_cnn_cat  = to_categorical(y_te_cnn,  num_classes=2)

print(f"DNN  - Train: {X_tr_dnn.shape}, Test: {X_te_dnn.shape}")
print(f"CNN  - Train: {X_tr_cnn.shape}, Test: {X_te_cnn.shape}")

# ============================================================
# BƯỚC 5: MÔ HÌNH DNN (Sequential Deep Neural Network)
# ============================================================
print("\n" + "=" * 60)
print("BUOC 5: XAY DUNG VA HUAN LUYEN DNN")
print("=" * 60)

def build_dnn(input_dim):
    """
    Kiến trúc DNN theo bài báo:
    Input(8) -> Dense(400,ReLU) -> Dropout(0.2)
             -> Dense(800,ReLU) -> Dropout(0.2)
             -> Dense(800,ReLU) -> Dropout(0.2)
             -> Dense(400,ReLU)
             -> Output(2, Softmax)
    """
    model = Sequential(name='Sequential_DNN')
    model.add(Dense(400, input_dim=input_dim, activation='relu',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

dnn_model = build_dnn(X_tr_dnn.shape[1])
dnn_model.summary()

print("\nHuan luyen DNN (100 epochs)...")
dnn_history = dnn_model.fit(
    X_tr_dnn, y_tr_cat,
    epochs=100,
    batch_size=50,
    validation_split=0.33,
    verbose=0   # Đặt verbose=1 nếu muốn xem từng epoch
)
print(f"DNN - Final val_accuracy: {dnn_history.history['val_accuracy'][-1]:.4f}")

# ============================================================
# BƯỚC 6: MÔ HÌNH 1D-CNN
# ============================================================
print("\n" + "=" * 60)
print("BUOC 6: XAY DUNG VA HUAN LUYEN 1D-CNN")
print("=" * 60)

# Reshape: (samples, features, 1) cho Conv1D
X_tr_cnn_r = X_tr_cnn.reshape(X_tr_cnn.shape[0], X_tr_cnn.shape[1], 1)
X_te_cnn_r = X_te_cnn.reshape(X_te_cnn.shape[0], X_te_cnn.shape[1], 1)

def build_cnn(input_shape):
    """
    Kiến trúc 1D-CNN:
    Conv1D(64) -> BN -> MaxPool -> Dropout
    Conv1D(128) -> BN -> MaxPool -> Dropout
    Flatten -> Dense(256) -> Dense(128) -> Output(2)
    """
    model = Sequential(name='1D_CNN')
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

cnn_model = build_cnn((X_tr_cnn_r.shape[1], 1))
cnn_model.summary()

print("\nHuan luyen 1D-CNN (100 epochs)...")
cnn_history = cnn_model.fit(
    X_tr_cnn_r, y_tr_cnn_cat,
    epochs=100,
    batch_size=50,
    validation_split=0.33,
    verbose=0
)
print(f"CNN - Final val_accuracy: {cnn_history.history['val_accuracy'][-1]:.4f}")

# ============================================================
# BƯỚC 7: ĐÁNH GIÁ & SO SÁNH
# ============================================================
print("\n" + "=" * 60)
print("BUOC 7: DANH GIA MO HINH")
print("=" * 60)

y_pred_dnn = np.argmax(dnn_model.predict(X_te_dnn), axis=1)
y_pred_cnn = np.argmax(cnn_model.predict(X_te_cnn_r), axis=1)

print("\n--- DNN (8 features) ---")
print(classification_report(y_te, y_pred_dnn, target_names=['Normal', 'Attack']))

print("--- 1D-CNN (43 features) ---")
print(classification_report(y_te_cnn, y_pred_cnn, target_names=['Normal', 'Attack']))

# Bảng so sánh
results = {
    'Model'    : ['DNN (8 features)', '1D-CNN (43 features)'],
    'Accuracy' : [accuracy_score(y_te, y_pred_dnn),
                  accuracy_score(y_te_cnn, y_pred_cnn)],
    'Precision': [precision_score(y_te, y_pred_dnn, average='weighted'),
                  precision_score(y_te_cnn, y_pred_cnn, average='weighted')],
    'Recall'   : [recall_score(y_te, y_pred_dnn, average='weighted'),
                  recall_score(y_te_cnn, y_pred_cnn, average='weighted')],
    'F1-Score' : [f1_score(y_te, y_pred_dnn, average='weighted'),
                  f1_score(y_te_cnn, y_pred_cnn, average='weighted')],
}
results_df = pd.DataFrame(results)
print("\n--- BANG SO SANH ---")
print(results_df.to_string(index=False))

# ============================================================
# BƯỚC 8: BIỂU ĐỒ KẾT QUẢ
# ============================================================
print("\n" + "=" * 60)
print("BUOC 8: VE BIEU DO")
print("=" * 60)

# (A) Learning Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Learning Curves - DNN vs 1D-CNN', fontsize=16, fontweight='bold')

axes[0,0].plot(dnn_history.history['accuracy'],     label='Train', color='#1565C0')
axes[0,0].plot(dnn_history.history['val_accuracy'], label='Validation', color='#FB8C00')
axes[0,0].set_title('DNN – Accuracy'); axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(dnn_history.history['loss'],     label='Train', color='#1565C0')
axes[0,1].plot(dnn_history.history['val_loss'], label='Validation', color='#FB8C00')
axes[0,1].set_title('DNN – Loss'); axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[1,0].plot(cnn_history.history['accuracy'],     label='Train', color='#2E7D32')
axes[1,0].plot(cnn_history.history['val_accuracy'], label='Validation', color='#C62828')
axes[1,0].set_title('1D-CNN – Accuracy'); axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Accuracy'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(cnn_history.history['loss'],     label='Train', color='#2E7D32')
axes[1,1].plot(cnn_history.history['val_loss'], label='Validation', color='#C62828')
axes[1,1].set_title('1D-CNN – Loss'); axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Loss'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Luu: learning_curves.png")

# (B) Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, title in [
    (axes[0], y_te,     y_pred_dnn, f'DNN – Acc={accuracy_score(y_te, y_pred_dnn):.4f}'),
    (axes[1], y_te_cnn, y_pred_cnn, f'1D-CNN – Acc={accuracy_score(y_te_cnn, y_pred_cnn):.4f}'),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'])
    ax.set_title(title, fontsize=12); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Luu: confusion_matrix.png")

# (C) ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
y_prob_dnn = dnn_model.predict(X_te_dnn)[:,1]
y_prob_cnn = cnn_model.predict(X_te_cnn_r)[:,1]
for ax, y_true, y_prob, title, color in [
    (axes[0], y_te,     y_prob_dnn, 'DNN (8 features)',    '#1565C0'),
    (axes[1], y_te_cnn, y_prob_cnn, '1D-CNN (43 features)','#2E7D32'),
]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.suptitle('ROC Curves', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Luu: roc_curves.png")

# (D) Bar Chart so sánh
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dnn_scores = [results['Accuracy'][0], results['Precision'][0],
              results['Recall'][0],    results['F1-Score'][0]]
cnn_scores = [results['Accuracy'][1], results['Precision'][1],
              results['Recall'][1],    results['F1-Score'][1]]

x   = np.arange(len(metrics_names))
w   = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
b1  = ax.bar(x - w/2, dnn_scores, w, label='DNN (8 features)', color='#1565C0', alpha=0.85)
b2  = ax.bar(x + w/2, cnn_scores, w, label='1D-CNN (43 features)', color='#2E7D32', alpha=0.85)
for bar in [*b1, *b2]:
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title('So sanh DNN vs 1D-CNN tren UNSW-NB15', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Luu: model_comparison.png")

print("\n" + "=" * 60)
print("HOAN THANH! Cac file bien do da duoc luu.")
print("Cac file output: feature_importance.png, learning_curves.png,")
print("                 confusion_matrix.png, roc_curves.png, model_comparison.png")
print("=" * 60)