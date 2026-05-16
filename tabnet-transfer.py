import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

# =====================
# 1. 读取训练数据（QLD）
# =====================
feature_columns = [
    "CTI", "SPI", "DTG", "ETa_mean_dry", "ETa_mean_annual",
    "clay_mean", "cv_lst", "elevation", "mTPI", "msavi",
    "ndvi", "ndwi_leaf", "ndwi_water", "pr_mean_dry", "pr_mean_annual", "wtd_2015"
]
target_column = "class2"

qld = pd.read_csv('dataset/NSW_Terrestrialfinal.csv')
qld = qld.dropna(subset=feature_columns + [target_column])
qld = qld[qld[target_column].isin([0, 1])].reset_index(drop=True)
# =====================
# 2. 读取外部测试集
# =====================
test_datasets = {
    "QLD": "dataset/QLD_Terrestrialfinal.csv",
    "VIC": "dataset/VIC_Terrestrialfinal.csv",
    "WA": "dataset/WA_Terrestrialfinal.csv",
    "SA": "dataset/SA_Terrestrialfinal.csv"

}
test_data = {}
for name, path in test_datasets.items():
    df = pd.read_csv(path)
    df = df.dropna(subset=feature_columns + [target_column])
    df = df[df[target_column].isin([0, 1])].reset_index(drop=True)
    test_data[name] = df

# =====================
# 3. 全局归一化（在折前）
# =====================
scaler = MinMaxScaler().fit(qld[feature_columns])

# transform training data (global)
X_qld = scaler.transform(qld[feature_columns])
y_qld = qld[target_column].values.astype(np.int64)  # ensure int

# transform all test sets with the same global scaler
test_sets = {}
for name, df in test_data.items():
    X_test = scaler.transform(df[feature_columns])
    y_test = df[target_column].values.astype(np.int64)
    test_sets[name] = (X_test, y_test)

# =====================
# 4. TabNet 设置（按你提供的超参数）
# =====================
# choose device_name safely
device_name = "cuda" if torch.cuda.is_available() else "cpu"

tabnet_base_params = dict(
    n_d=16, n_a=16,
    mask_type='entmax',
    device_name=device_name
)

# =====================
# 5. 交叉验证设置（外层 5 折）
# =====================
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

outer_metrics_qld, outer_conf_qld = [], []
outer_metrics_tests = {name: [] for name in test_sets}
outer_conf_tests = {name: [] for name in test_sets}

fold_num = 1
for train_idx, val_idx in outer_cv.split(X_qld, y_qld):
    print(f"\n=== 折 {fold_num} ===")

    # 训练/验证切分（已经全局归一化）
    X_train, X_val = X_qld[train_idx], X_qld[val_idx]
    y_train, y_val = y_qld[train_idx], y_qld[val_idx]

    # -----------------
    # 构建 TabNet 模型
    # -----------------
    tabnet_clf = TabNetClassifier(**tabnet_base_params)

    # 训练（按你的要求：不做 early stop，直接跑 60 轮）
    tabnet_clf.fit(
        X_train, y_train,
        max_epochs=80,
        patience=0,
        # 不传 patience，保证不会早停；按你说直接跑60轮
        batch_size=10240,
        virtual_batch_size=1024,
        num_workers=0,
        drop_last=True
    )
    print(f"模型已训练完")
    # 外部测试集（使用预先全局归一化的 test_sets）
    for name, (X_test, y_test) in test_sets.items():
        pred_test_prob = tabnet_clf.predict_proba(X_test)
        pred_test = (pred_test_prob[:, 1] > 0.5).astype(int)
        acc = accuracy_score(y_test, pred_test)
        auc = roc_auc_score(y_test, pred_test_prob[:, 1])
        cm = confusion_matrix(y_test, pred_test)
        print(f"{name} 测试 Acc: {acc:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_test, pred_test, digits=4))
        outer_metrics_tests[name].append({'acc': acc, 'auc': auc})
        outer_conf_tests[name].append(cm)
    # 验证（QLD 的验证集）
    pred_val_prob = tabnet_clf.predict_proba(X_val)
    pred_val = (pred_val_prob[:, 1] > 0.5).astype(int)
    acc_qld = accuracy_score(y_val, pred_val)
    auc_qld = roc_auc_score(y_val, pred_val_prob[:, 1])
    cm_qld = confusion_matrix(y_val, pred_val)
    print(f"QLD 验证 Acc: {acc_qld:.4f}, AUC: {auc_qld:.4f}")
    print(classification_report(y_val, pred_val, digits=4))
    outer_metrics_qld.append({'acc': acc_qld, 'auc': auc_qld})
    outer_conf_qld.append(cm_qld)

    fold_num += 1

# =====================
# 6. 汇总结果（替换原来的汇总段）
# =====================

for name in test_sets:
    print(f"\n=== {name} 总体 ===")
    print("累加混淆矩阵:\n", sum(outer_conf_tests[name]))
    print(f"{name} 平均 Acc: {np.mean([m['acc'] for m in outer_metrics_tests[name]]):.4f}")
    print(f"{name} 平均 AUC: {np.mean([m['auc'] for m in outer_metrics_tests[name]]):.4f}")
    print(f"{name} 每折 AUC 列表: {[m['auc'] for m in outer_metrics_tests[name]]}")
    print("\n=== QLD 总体 ===")
print("累加混淆矩阵:\n", sum(outer_conf_qld))
print("平均 Acc:", np.mean([m['acc'] for m in outer_metrics_qld]))
print("平均 AUC:", np.mean([m['auc'] for m in outer_metrics_qld]))
print("每折 AUC 列表:", [m['auc'] for m in outer_metrics_qld])

