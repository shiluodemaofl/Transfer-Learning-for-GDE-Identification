import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    "VIC": "dataset/VIC_Terrestrialfinal.csv",
    "SA": "dataset/SA_Terrestrialfinal.csv",
    "WA": "dataset/WA_Terrestrialfinal.csv",
    "QLD": "dataset/QLD_Terrestrialfinal.csv"
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
y_qld = qld[target_column].values

# transform all test sets with the same global scaler
test_sets = {}
for name, df in test_data.items():
    X_test = scaler.transform(df[feature_columns])
    y_test = df[target_column].values
    test_sets[name] = (X_test, y_test)

# =====================
# 4. 定义 MLP 模型 & 工具函数
# =====================
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 204),
            nn.ReLU(),
            nn.Linear(204, 102),
            nn.ReLU(),
            nn.Linear(102, 51),
            nn.ReLU(),
            nn.Linear(51, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, epochs=60):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_prob)

# =====================
# 5. 交叉验证设置
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X_qld.shape[1]
num_classes = len(np.unique(y_qld))

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

outer_metrics_qld, outer_conf_qld = [], []
outer_metrics_tests = {name: [] for name in test_sets}
outer_conf_tests = {name: [] for name in test_sets}

fold_num = 1
for train_idx, val_idx in outer_cv.split(X_qld, y_qld):
    print(f"\n=== 折 {fold_num} ===")

    # 直接使用已经全局归一化过的 X_qld
    X_train, X_val = X_qld[train_idx], X_qld[val_idx]
    y_train, y_val = y_qld[train_idx], y_qld[val_idx]

    # 转换为 Tensor + DataLoader（放在 device）
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=512, shuffle=False)

    # 定义模型、损失和优化器
    model = MLP(n_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0025)

    # 训练
    train_model(model, train_loader, criterion, optimizer, epochs=90)

    # 验证（QLD 的验证集）
    y_true_val, y_prob_val = evaluate_model(model, val_loader, device)
    y_pred_val = np.argmax(y_prob_val, axis=1)
    acc_qld = accuracy_score(y_true_val, y_pred_val)
    auc_qld = roc_auc_score(y_true_val, y_prob_val[:, 1])
    cm_qld = confusion_matrix(y_true_val, y_pred_val)
    print(f"QLD 验证 Acc: {acc_qld:.4f}, AUC: {auc_qld:.4f}")
    print(classification_report(y_true_val, y_pred_val, digits=4))
    outer_metrics_qld.append({'acc': acc_qld, 'auc': auc_qld})
    outer_conf_qld.append(cm_qld)

    # 外部测试集（直接使用上面已全局归一化的 test_sets）
    for name, (X_test, y_test) in test_sets.items():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=512, shuffle=False)

        y_true_test, y_prob_test = evaluate_model(model, test_loader, device)
        y_pred_test = np.argmax(y_prob_test, axis=1)
        acc = accuracy_score(y_true_test, y_pred_test)
        auc = roc_auc_score(y_true_test, y_prob_test[:, 1])
        cm = confusion_matrix(y_true_test, y_pred_test)
        print(f"{name} 测试 Acc: {acc:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_true_test, y_pred_test, digits=4))
        outer_metrics_tests[name].append({'acc': acc, 'auc': auc})
        outer_conf_tests[name].append(cm)

    fold_num += 1

# =====================
# 6. 汇总结果
# =====================
print("\n=== QLD 总体 ===")
print("累加混淆矩阵:\n", sum(outer_conf_qld))
print("平均 Acc:", np.mean([m['acc'] for m in outer_metrics_qld]))
print("平均 AUC:", np.mean([m['auc'] for m in outer_metrics_qld]))

for name in test_sets:
    print(f"\n=== {name} 总体 ===")
    print("累加混淆矩阵:\n", sum(outer_conf_tests[name]))
    print("平均 Acc:", np.mean([m['acc'] for m in outer_metrics_tests[name]]))
    print("平均 AUC:", np.mean([m['auc'] for m in outer_metrics_tests[name]]))

