import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
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
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

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
    "QLD": "dataset/QLD_Terrestrialfinal.csv",
    "VIC": "dataset/VIC_Terrestrialfinal.csv",
    "WA":  "dataset/WA_Terrestrialfinal.csv",
    "SA":  "dataset/SA_Terrestrialfinal.csv"
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

outer_metrics_qld, outer_conf_qld = [], []
# per-target containers
per_target_confs = {name: [] for name in test_data_scaled.keys()}
per_target_metrics = {name: [] for name in test_data_scaled.keys()}

fold_num = 1
for train_idx, val_idx in outer_cv.split(X_qld, y_qld):
    print(f"\n=== 折 {fold_num} ===")
    X_train, X_val = X_qld[train_idx], X_qld[val_idx]
    y_train, y_val = y_qld[train_idx], y_qld[val_idx]

    # 3. 内层目标函数（以 validation 的 binary_logloss 最小化为目标 -> Bayesian 最大化负值）
    def inner_objective(n_estimators, max_depth, learning_rate,
                        bagging_fraction, feature_fraction, early_stop, max_leaf_nodes):
        # cast
        n_estimators = int(round(n_estimators))
        max_depth = int(round(max_depth))
        max_leaf_nodes = int(round(max_leaf_nodes))
        early_stop = int(min(round(early_stop), max(1, n_estimators - 1)))

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': float(learning_rate),
            'bagging_fraction': float(bagging_fraction),
            'feature_fraction': float(feature_fraction),
            'max_leaf_nodes': max_leaf_nodes,
            'random_state': RANDOM_SEED,
            'metric': 'binary_logloss',
            'verbosity': -1
        }
        clf = lgb.LGBMClassifier(**params)
        # 用外层的 X_train/X_val 作内层验证（与你原脚本一致）
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]
        )
        # 返回 -logloss 以便 BayesianOptimization maximize
        return -clf.best_score_['valid_0']['binary_logloss']

    # 4. 贝叶斯优化
    pbounds = {
       'n_estimators': (1, 2000),
        'max_depth': (1, 10),
        'learning_rate': (0.01, 0.4),
        'bagging_fraction': (0.1, 1.0),
        'feature_fraction': (0.1, 1.0),
        'early_stop': (5, 50),
        'max_leaf_nodes': (100, 3000)
    }
    optimizer = BayesianOptimization(f=inner_objective, pbounds=pbounds, random_state=RANDOM_SEED)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max['params']
    best_inner = {
        'n_estimators': int(round(best['n_estimators'])),
        'max_depth': int(round(best['max_depth'])),
        'learning_rate': float(best['learning_rate']),
        'bagging_fraction': float(best['bagging_fraction']),
        'feature_fraction': float(best['feature_fraction']),
        'early_stop': int(min(round(best['early_stop']), max(1, int(round(best['n_estimators'])) - 1))),
        'max_leaf_nodes': int(round(best['max_leaf_nodes']))
    }
    print("内层最优参数：", best_inner)

    # 5. 用最优参数训练最终模型
    clf_final = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=best_inner['n_estimators'],
        max_depth=best_inner['max_depth'],
        learning_rate=best_inner['learning_rate'],
        bagging_fraction=best_inner['bagging_fraction'],
        feature_fraction=best_inner['feature_fraction'],
        max_leaf_nodes=best_inner['max_leaf_nodes'],
        random_state=RANDOM_SEED,
        metric='binary_logloss',
        verbosity=-1
    )
    clf_final.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(best_inner['early_stop'], verbose=False)]
    )

    # QLD 验证（折内外部验证）
    num_iter = getattr(clf_final, "best_iteration_", None)
    if num_iter is None:
        pred_val_prob = clf_final.predict_proba(X_val)
    else:
        pred_val_prob = clf_final.predict_proba(X_val, num_iteration=num_iter)
    pred_val = np.argmax(pred_val_prob, axis=1)
    acc_qld = accuracy_score(y_val, pred_val)
    auc_qld = roc_auc_score(y_val, pred_val_prob[:, 1])
    cm_qld = confusion_matrix(y_val, pred_val)
    print(f"QLD 验证 Acc: {acc_qld:.4f}, AUC: {auc_qld:.4f}")
    print(classification_report(y_val, pred_val, digits=4))
    outer_metrics_qld.append({'acc': acc_qld, 'auc': auc_qld})
    outer_conf_qld.append(cm_qld)

    # 对每个目标域分别评估
    for name, (X_t, y_t) in test_data_scaled.items():
        if num_iter is None:
            pred_t_prob = clf_final.predict_proba(X_t)
        else:
            pred_t_prob = clf_final.predict_proba(X_t, num_iteration=num_iter)
        pred_t = np.argmax(pred_t_prob, axis=1)
        acc_t = accuracy_score(y_t, pred_t)
        auc_t = roc_auc_score(y_t, pred_t_prob[:, 1])
        cm_t = confusion_matrix(y_t, pred_t)
        print(f"{name} 测试 Acc: {acc_t:.4f}, AUC: {auc_t:.4f}")
        # 可选：print(classification_report(y_t, pred_t, digits=4))

        per_target_confs[name].append(cm_t)
        per_target_metrics[name].append({'acc': acc_t, 'auc': auc_t})

    fold_num += 1

# ===================== 汇总结果 =====================
print("\n=== QLD 总体 ===")
print("累加混淆矩阵:\n", np.sum(np.array(outer_conf_qld), axis=0))
print("平均 Acc:", np.mean([m['acc'] for m in outer_metrics_qld]))
print("平均 AUC:", np.mean([m['auc'] for m in outer_metrics_qld]))

for name in test_data_scaled.keys():
    confs = per_target_confs[name]
    if len(confs) == 0:
        continue
    sum_conf = np.sum(np.array(confs), axis=0)
    mean_acc = np.mean([m['acc'] for m in per_target_metrics[name]])
    mean_auc = np.mean([m['auc'] for m in per_target_metrics[name]])
    print(f"\n=== {name} 汇总（按折累加） ===")
    print("累加混淆矩阵:\n", sum_conf)
    print(f"{name} 平均 Acc: {mean_acc:.4f}, 平均 AUC: {mean_auc:.4f}")
