import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization

# 1. 读取训练数据（NSW）
feature_columns = [
    "CTI", "SPI", "DTG", "ETa_mean_dry", "ETa_mean_annual",
    "clay_mean", "cv_lst", "elevation", "mTPI", "msavi",
    "ndvi", "ndwi_leaf", "ndwi_water", "pr_mean_dry", "pr_mean_annual", "wtd_2015"
]
target_column = "class2"

qld = pd.read_csv('dataset/NSW_Terrestrialfinal.csv')
qld = qld.dropna(subset=feature_columns + [target_column])
qld = qld[qld[target_column].isin([0, 1])].reset_index(drop=True)

# 2. 读取外部测试集
test_datasets = {
    "VIC": "dataset/VIC_Terrestrialfinal.csv",
    "SA":  "dataset/SA_Terrestrialfinal.csv",
    "WA":  "dataset/WA_Terrestrialfinal.csv",
    "QLD": "dataset/QLD_Terrestrialfinal.csv"
}

# 清洗并存入字典
test_data = {}
for name, path in test_datasets.items():
    df = pd.read_csv(path)
    df = df.dropna(subset=feature_columns + [target_column])
    df = df[df[target_column].isin([0, 1])].reset_index(drop=True)
    test_data[name] = df

# ===================== 全局归一化 =====================
scaler = MinMaxScaler().fit(qld[feature_columns])

X_qld = scaler.transform(qld[feature_columns])
y_qld = qld[target_column].values

test_sets = {}
for name, df in test_data.items():
    X_test = scaler.transform(df[feature_columns])
    y_test = df[target_column].values
    test_sets[name] = (X_test, y_test)
# ====================================================

# 3. 外层 5 折
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

# containers for summary
outer_metrics_qld = []          # list of dicts {'acc':..., 'auc':...}
outer_conf_qld = []
outer_metrics_tests = {name: [] for name in test_sets}
outer_conf_tests = {name: [] for name in test_sets}

fold_num = 1
for train_idx, val_idx in outer_cv.split(X_qld, y_qld):
    print(f"\n=== 折 {fold_num} ===")
    X_train, X_val = X_qld[train_idx], X_qld[val_idx]
    y_train, y_val = y_qld[train_idx], y_qld[val_idx]

    # 内层目标函数
    def inner_objective(depth, learning_rate, iterations, early_stop, l2_leaf_reg, bagging_temperature):
        depth = int(depth)
        iterations = int(iterations)
        early_stop = int(min(early_stop, iterations - 1))
        l2_leaf_reg = float(l2_leaf_reg)
        bagging_temperature = float(bagging_temperature)

        model = CatBoostClassifier(
            objective="Logloss",
            eval_metric="Logloss",
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            od_type="Iter",
            od_wait=early_stop,
            random_seed=2025,
            verbose=False,
            task_type="GPU"
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return -model.best_score_["validation"]["Logloss"]

    # 贝叶斯优化
    pbounds = {
        'depth': (1, 10),
        'learning_rate': (0.01, 0.4),
        'iterations': (1, 2000),
        'early_stop': (5, 50),
        'l2_leaf_reg': (1, 10),
        'bagging_temperature': (0.1, 1.0)
    }
    optimizer = BayesianOptimization(f=inner_objective, pbounds=pbounds, random_state=2025)
    optimizer.maximize(init_points=5, n_iter=15)

    best = optimizer.max['params']
    best_inner = {
        'depth': int(best['depth']),
        'learning_rate': best['learning_rate'],
        'iterations': int(best['iterations']),
        'early_stop': int(min(best['early_stop'], int(best['iterations']) - 1)),
        'l2_leaf_reg': best['l2_leaf_reg'],
        'bagging_temperature': best['bagging_temperature']
    }
    print("内层最优参数：", best_inner)

    # 训练最终模型
    model = CatBoostClassifier(
        objective="Logloss",
        eval_metric="Logloss",
        depth=best_inner['depth'],
        learning_rate=best_inner['learning_rate'],
        iterations=best_inner['iterations'],
        l2_leaf_reg=best_inner['l2_leaf_reg'],
        bagging_temperature=best_inner['bagging_temperature'],
        od_type="Iter",
        od_wait=best_inner['early_stop'],
        random_seed=2025,
        verbose=False,
        task_type="GPU"
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    # NSW (验证集)
    pred_val_prob = model.predict_proba(X_val)[:, 1]
    pred_val = (pred_val_prob > 0.5).astype(int)
    acc_qld = accuracy_score(y_val, pred_val)
    try:
        auc_qld = roc_auc_score(y_val, pred_val_prob)
    except Exception:
        auc_qld = float('nan')
    cm_qld = confusion_matrix(y_val, pred_val)
    print(f"NSW 验证 Acc: {acc_qld:.4f}, AUC: {auc_qld:.4f}" if not np.isnan(auc_qld) else f"NSW 验证 Acc: {acc_qld:.4f}, AUC: N/A")
    print(classification_report(y_val, pred_val, digits=4))
    outer_metrics_qld.append({'acc': acc_qld, 'auc': auc_qld})
    outer_conf_qld.append(cm_qld)

    # 多个测试集循环评估
    for name, (X_test, y_test) in test_sets.items():
        pred_prob = model.predict_proba(X_test)[:, 1]
        pred = (pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, pred_prob)
        except Exception:
            auc = float('nan')
        cm = confusion_matrix(y_test, pred)
        print(f"{name} 测试  Acc: {acc:.4f}, AUC: {auc:.4f}" if not np.isnan(auc) else f"{name} 测试  Acc: {acc:.4f}, AUC: N/A")
        print(classification_report(y_test, pred, digits=4))
        outer_metrics_tests[name].append({'acc': acc, 'auc': auc})
        outer_conf_tests[name].append(cm)

    fold_num += 1

# 结果汇总
print("\n=== NSW（源域）总体 ===")
sum_cm_qld = np.sum(np.array(outer_conf_qld), axis=0)
mean_acc_qld = np.mean([m['acc'] for m in outer_metrics_qld])
auc_list_qld = [m['auc'] for m in outer_metrics_qld if not np.isnan(m['auc'])]
mean_auc_qld = np.mean(auc_list_qld) if len(auc_list_qld) > 0 else float('nan')
print("累加混淆矩阵:\n", sum_cm_qld)
print("平均 Acc:", mean_acc_qld)
print("平均 AUC:", mean_auc_qld if not np.isnan(mean_auc_qld) else "N/A")

for name in test_sets:
    print(f"\n=== {name} 总体 ===")
    sum_cm = np.sum(np.array(outer_conf_tests[name]), axis=0)
    mean_acc = np.mean([m['acc'] for m in outer_metrics_tests[name]])
    auc_list = [m['auc'] for m in outer_metrics_tests[name] if not np.isnan(m['auc'])]
    mean_auc = np.mean(auc_list) if len(auc_list) > 0 else float('nan')
    print("累加混淆矩阵:\n", sum_cm)
    print(f"{name} 平均 Acc: {mean_acc}")
    print(f"{name} 平均 AUC: {mean_auc if not np.isnan(mean_auc) else 'N/A'}")
