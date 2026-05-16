import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from bayes_opt import BayesianOptimization

# ------------------ 配置 ------------------
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

feature_columns = [
    "CTI", "SPI", "DTG", "ETa_mean_dry", "ETa_mean_annual",
    "clay_mean", "cv_lst", "elevation", "mTPI", "msavi",
    "ndvi", "ndwi_leaf", "ndwi_water", "pr_mean_dry", "pr_mean_annual", "wtd_2015"
]
target_column = "class2"

# 外层 CV 参数
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# 贝叶斯优化参数（可调）
init_points = 5
n_iter = 15
# -------------------------------------------

# 1) 读取源域 QLD（训练源）
qld = pd.read_csv('dataset/NSW_Terrestrialfinal.csv')
qld = qld.dropna(subset=feature_columns + [target_column])
qld = qld[qld[target_column].isin([0, 1])].reset_index(drop=True)

# 2) 读取多个目标域（test datasets）
test_datasets = {
    "QLD":  "dataset/QLD_Terrestrialfinal.csv",
    "VIC":  "dataset/VIC_Terrestrialfinal.csv",
    "WA":   "dataset/WA_Terrestrialfinal.csv",
    "SA":   "dataset/SA_Terrestrialfinal.csv"
}
test_data = {}
for name, path in test_datasets.items():
    df = pd.read_csv(path)
    df = df.dropna(subset=feature_columns + [target_column])
    df = df[df[target_column].isin([0, 1])].reset_index(drop=True)
    test_data[name] = df

# ===================== 全局归一化（只用 QLD 拟合 scaler） =====================
scaler = MinMaxScaler().fit(qld[feature_columns])

# transform QLD (source)
X_qld = scaler.transform(qld[feature_columns])
y_qld = qld[target_column].values

# transform all target datasets using same scaler and store
test_data_scaled = {}
for name, df in test_data.items():
    X_t = scaler.transform(df[feature_columns])
    y_t = df[target_column].values
    test_data_scaled[name] = (X_t, y_t)
# ====================================================

n_features = X_qld.shape[1]

# 随机森林的贝叶斯优化搜索空间（按你给的）
pbounds = {
    'n_estimators': (5, 50),
    'max_depth': (5, 30),
    'max_features': (1, n_features),
    'min_samples_leaf': (1, 10),
    'max_samples': (0.5, 1.0),
    'max_leaf_nodes': (100, 3000)
}

# 容器：为每个目标域保存每折混淆矩阵和指标
outer_conf_qld = []   # QLD 外部验证每折混淆矩阵
outer_metrics_qld = []  # QLD 每折指标
best_params_list = []

# per-target containers
per_target_confs = {name: [] for name in test_data_scaled.keys()}
per_target_metrics = {name: [] for name in test_data_scaled.keys()}

fold_num = 1
for train_idx, val_idx in outer_cv.split(X_qld, y_qld):
    print(f"\n=== 外部折 {fold_num} ===")
    X_train, X_val = X_qld[train_idx], X_qld[val_idx]
    y_train, y_val = y_qld[train_idx], y_qld[val_idx]

    # 在外层训练集（X_train）上再切出内层训练/验证用于调参
    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
    )

    # 内层目标函数：返回内部验证集 accuracy（贝叶斯要最大化这个）
    def inner_objective(n_estimators, max_depth, max_features, min_samples_leaf, max_samples, max_leaf_nodes):
        # cast / clip to valid types
        n_estimators = int(round(n_estimators))
        max_depth = int(round(max_depth))
        max_features = int(round(max_features))
        min_samples_leaf = int(round(min_samples_leaf))
        max_leaf_nodes = int(round(max_leaf_nodes))

        max_features = max(1, min(max_features, n_features))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=True,
            max_samples=float(max_samples),
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        clf.fit(X_inner_train, y_inner_train)
        preds = clf.predict(X_inner_val)
        return accuracy_score(y_inner_val, preds)

    optimizer = BayesianOptimization(
        f=inner_objective,
        pbounds=pbounds,
        random_state=RANDOM_SEED,
        verbose=0
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # 最佳参数（转换类型）
    best = optimizer.max['params']
    best_params = {
        'n_estimators': int(round(best['n_estimators'])),
        'max_depth': int(round(best['max_depth'])),
        'max_features': int(round(best['max_features'])),
        'min_samples_leaf': int(round(best['min_samples_leaf'])),
        'max_samples': float(best['max_samples']),
        'max_leaf_nodes': int(round(best['max_leaf_nodes']))
    }
    best_params['max_features'] = max(1, min(best_params['max_features'], n_features))
    print("内层最优参数：", best_params)
    best_params_list.append(best_params)

    # 用外部训练集全部数据训练最终模型（注意：X_train 已经全局归一化）
    clf_final = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_leaf_nodes=best_params['max_leaf_nodes'],
        bootstrap=True,
        max_samples=best_params['max_samples'],
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    clf_final.fit(X_train, y_train)

    # 在外部验证集上评估（QLD 外部验证）
    y_val_pred = clf_final.predict(X_val)
    y_val_prob = clf_final.predict_proba(X_val)[:, 1] if hasattr(clf_final, "predict_proba") else None
    acc_val = accuracy_score(y_val, y_val_pred)
    auc_val = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
    cm_val = confusion_matrix(y_val, y_val_pred)
    print(f"QLD 外部验证 Acc: {acc_val:.4f}, AUC: {auc_val if not np.isnan(auc_val) else 'N/A'}")
    print(classification_report(y_val, y_val_pred, digits=4))

    outer_conf_qld.append(cm_val)
    outer_metrics_qld.append({'acc': acc_val, 'auc': auc_val})

    # 对每个目标域分别评估
    for name, (X_t, y_t) in test_data_scaled.items():
        y_t_pred = clf_final.predict(X_t)
        y_t_prob = clf_final.predict_proba(X_t)[:, 1] if hasattr(clf_final, "predict_proba") else None
        acc_t = accuracy_score(y_t, y_t_pred)
        auc_t = roc_auc_score(y_t, y_t_prob) if y_t_prob is not None else np.nan
        cm_t = confusion_matrix(y_t, y_t_pred)
        print(f"{name} 测试 Acc: {acc_t:.4f}, AUC: {auc_t if not np.isnan(auc_t) else 'N/A'}")
        # 可选：打印 classification_report 每折
        # print(classification_report(y_t, y_t_pred, digits=4))

        per_target_confs[name].append(cm_t)
        per_target_metrics[name].append({'acc': acc_t, 'auc': auc_t})

    fold_num += 1

# ========== 汇总并打印 ==========
# QLD 汇总
all_conf_qld = np.sum(np.array(outer_conf_qld), axis=0)
mean_acc_qld = np.mean([m['acc'] for m in outer_metrics_qld])
mean_auc_qld = np.mean([m['auc'] for m in outer_metrics_qld if not np.isnan(m['auc'])]) if len([m for m in outer_metrics_qld if not np.isnan(m['auc'])])>0 else np.nan
print("\n=== QLD 汇总（按折累加） ===")
print("累加混淆矩阵 (QLD 外部验证 每折累加):\n", all_conf_qld)
print("QLD 平均 Acc:", mean_acc_qld, "QLD 平均 AUC (可用时):", mean_auc_qld)

# 每个目标域汇总
for name in test_data_scaled.keys():
    confs = per_target_confs[name]
    if len(confs) == 0:
        continue
    sum_conf = np.sum(np.array(confs), axis=0)
    mean_acc = np.mean([m['acc'] for m in per_target_metrics[name]])
    aucs = [m['auc'] for m in per_target_metrics[name] if not np.isnan(m['auc'])]
    mean_auc = np.mean(aucs) if len(aucs) > 0 else np.nan
    print(f"\n=== {name} 汇总（按折累加） ===")
    print("累加混淆矩阵:\n", sum_conf)
    print(f"{name} 平均 Acc: {mean_acc:.4f}, 平均 AUC: {mean_auc if not np.isnan(mean_auc) else 'N/A'}")

print("\n每折最佳参数：")
for i, p in enumerate(best_params_list, start=1):
    print(f"Fold {i}: {p}")
