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
import random
import os

# ---------------------------
# 可配置项（按需调整）
# ---------------------------
SEED = 2025
BATCH_SIZE = 512
EPOCHS = 90
PRETRAIN_EPOCHS = 5  # 前几轮只做 supervised
LR_MAIN = 2.5e-3
LR_DOMAIN = 2.5e-3
DOMAIN_LOSS_WEIGHT = 1.0
EPS = 1e-8

# 固定随机种子，尽量可复现
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# =====================
# 1. 读取训练数据（QLD）
# =====================
feature_columns = [
    "CTI", "SPI", "DTG", "ETa_mean_dry", "ETa_mean_annual",
    "clay_mean", "cv_lst", "elevation", "mTPI", "msavi",
    "ndvi", "ndwi_leaf", "ndwi_water", "pr_mean_dry", "pr_mean_annual", "wtd_2015"
]
target_column = "class2"

qld = pd.read_csv('dataset/NSW_Aquaticfinal.csv')
qld = qld.dropna(subset=feature_columns + [target_column])
qld = qld[qld[target_column].isin([0, 1])].reset_index(drop=True)

# =====================
# 2. 读取外部测试集（WA）
# =====================
sa = pd.read_csv('dataset/VIC_Aquaticfinal.csv')
sa = sa.dropna(subset=feature_columns + [target_column])
sa = sa[sa[target_column].isin([0, 1])].reset_index(drop=True)

# =====================
# 3. 全局归一化（在折前）
# =====================
scaler = MinMaxScaler().fit(qld[feature_columns])

# transform training data (global)
X_qld = scaler.transform(qld[feature_columns])
y_qld = qld[target_column].values

# transform WA test set with the same global scaler
X_vic = scaler.transform(sa[feature_columns])
y_vic = sa[target_column].values

# =====================
# 4. DAAN 所需模型（Feature, Classifier, MarginalDomainClf, ConditionalDomainClf）和 GRL
# =====================

class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer as autograd Function."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class FeatureExtractor(nn.Module):
    def __init__(self, input_size):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 204),
            nn.ReLU(),
            nn.Linear(204, 102),
            nn.ReLU(),
            nn.Linear(102, 51),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)  # 输出维度 51


class ClassifierHead(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, feat):
        return self.fc(feat)


class MarginalDomainClassifier(nn.Module):
    def __init__(self, feat_dim):
        super(MarginalDomainClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, feat):
        return self.net(feat)


class ConditionalDomainClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(ConditionalDomainClassifier, self).__init__()
        in_dim = feat_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, joint_feat):
        return self.net(joint_feat)

# =====================
# 5. 训练 / 评估工具函数（支持 DAAN 阶段）
# =====================

def train_epoch_supervised(feature_extractor, classifier, dataloader, criterion_cls, optimizer_main, device):
    feature_extractor.train()
    classifier.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer_main.zero_grad()
        feats = feature_extractor(X_batch)
        logits = classifier(feats)
        loss = criterion_cls(logits, y_batch)
        loss.backward()
        optimizer_main.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_epoch_daan(feature_extractor, classifier,
                     marginal_domain_clf, conditional_domain_clf,
                     source_loader, target_loader,
                     criterion_cls, criterion_domain,
                     optimizer_main, optimizer_domain,
                     device, grl_lambda, domain_loss_weight=1.0,
                     eps=EPS):
    """
    DAAN training on one epoch.
    - marginal_domain_clf: domain clf on features
    - conditional_domain_clf: domain clf on (feature concat class_prob)
    - grl_lambda: scalar for GRL
    - domain_loss_weight: global multiplier for domain alignment
    """
    feature_extractor.train()
    classifier.train()
    marginal_domain_clf.train()
    conditional_domain_clf.train()

    total_cls_loss = 0.0
    total_dom_loss = 0.0

    target_iter = iter(target_loader)
    for X_s, y_s in source_loader:
        X_s = X_s.to(device)
        y_s = y_s.to(device)

        # get a target batch (cycle if necessary)
        try:
            X_t = next(target_iter)[0]
        except StopIteration:
            target_iter = iter(target_loader)
            X_t = next(target_iter)[0]
        X_t = X_t.to(device)

        optimizer_main.zero_grad()
        optimizer_domain.zero_grad()

        # 1) forward source classification
        feats_s = feature_extractor(X_s)            # [bs, feat_dim]
        logits_s = classifier(feats_s)              # [bs, num_classes]
        loss_cls = criterion_cls(logits_s, y_s)

        # 2) compute marginal domain loss (features only)
        feats_t = feature_extractor(X_t)
        feats_concat = torch.cat([feats_s, feats_t], dim=0)  # [bs_s+bs_t, feat_dim]
        feats_rev_m = grad_reverse(feats_concat, lambd=grl_lambda)
        dom_logits_m = marginal_domain_clf(feats_rev_m)
        dom_labels = torch.cat([
            torch.zeros(feats_s.size(0), dtype=torch.long),
            torch.ones(feats_t.size(0), dtype=torch.long)
        ], dim=0).to(device)
        loss_dom_m = criterion_domain(dom_logits_m, dom_labels)

        # 3) compute conditional domain loss (feature concat with soft class probabilities)
        with torch.no_grad():
            probs_s = F.softmax(logits_s, dim=1)
        logits_t = classifier(feats_t)
        with torch.no_grad():
            probs_t = F.softmax(logits_t, dim=1)

        probs_s_det = probs_s.detach()
        probs_t_det = probs_t.detach()

        joint_s = torch.cat([feats_s, probs_s_det], dim=1)
        joint_t = torch.cat([feats_t, probs_t_det], dim=1)
        joint_concat = torch.cat([joint_s, joint_t], dim=0)  # [bs_s+bs_t, feat_dim + num_classes]

        joint_rev = grad_reverse(joint_concat, lambd=grl_lambda)
        dom_logits_c = conditional_domain_clf(joint_rev)
        loss_dom_c = criterion_domain(dom_logits_c, dom_labels)

        # 4) dynamic weight: balance marginal and conditional losses
        beta = (loss_dom_m / (loss_dom_m + loss_dom_c + eps)).detach()
        loss_dom = beta * loss_dom_m + (1.0 - beta) * loss_dom_c

        # 5) total loss: classification + domain_loss_weight * domain_loss
        loss = loss_cls + domain_loss_weight * loss_dom
        loss.backward()

        optimizer_main.step()
        optimizer_domain.step()

        total_cls_loss += loss_cls.item()
        total_dom_loss += loss_dom.item()

    n_steps = len(source_loader)
    return total_cls_loss / n_steps, total_dom_loss / n_steps


def evaluate_model(feature_extractor, classifier, loader, device):
    feature_extractor.eval()
    classifier.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = classifier(feature_extractor(X_batch))
            probs = F.softmax(outputs, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_prob)


# =====================
# 6. 交叉验证设置 + 训练主循环（在每折内实现 "前PRETRAIN_EPOCHS轮非 DAAN，之后加入 DAAN"）
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X_qld.shape[1]
num_classes = len(np.unique(y_qld))

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

outer_metrics_qld, outer_conf_qld = [], []
outer_metrics_tests = {"WA": []}
outer_conf_tests = {"WA": []}

# 将外部 WA 数据制作成 DataLoader（仅用于 domain 判别时作为“目标域无标签样本”以及在每折训练结束后的外部测试）
X_vic_tensor_all = torch.tensor(X_vic, dtype=torch.float32)
y_vic_tensor_all = torch.tensor(y_vic, dtype=torch.long)
target_domain_dataset = TensorDataset(X_vic_tensor_all)  # yields (X_t,) tuples

fold_num = 1
for train_idx, val_idx in outer_cv.split(X_qld, y_qld):
    print(f"\n=== 折 {fold_num} ===")

    # 使用已全局归一化过的 X_qld
    X_train, X_val = X_qld[train_idx], X_qld[val_idx]
    y_train, y_val = y_qld[train_idx], y_qld[val_idx]

    # 转换为 Tensor + DataLoader（放在 device 里）
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)

    target_loader = DataLoader(target_domain_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 定义模型、损失和优化器
    feat_dim = 51  # 如 FeatureExtractor 输出
    feature_extractor = FeatureExtractor(n_features).to(device)
    classifier = ClassifierHead(feat_dim, num_classes).to(device)
    marginal_domain_clf = MarginalDomainClassifier(feat_dim).to(device)
    conditional_domain_clf = ConditionalDomainClassifier(feat_dim, num_classes).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    optimizer_main = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=LR_MAIN)
    optimizer_domain = optim.Adam(list(marginal_domain_clf.parameters()) + list(conditional_domain_clf.parameters()), lr=LR_DOMAIN)

    # GRL lambda 调度（常见做法：随着训练进展从 0 -> 1 增大）
    def grl_lambda(progress):
        return 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0

    for epoch in range(EPOCHS):
        if epoch < PRETRAIN_EPOCHS:
            avg_sup_loss = train_epoch_supervised(
                feature_extractor, classifier, train_loader, criterion_cls, optimizer_main, device
            )
            print(f"Epoch [{epoch+1}/{EPOCHS}] SUP only, Loss: {avg_sup_loss:.4f}")
        else:
            progress = (epoch - PRETRAIN_EPOCHS) / max(1, (EPOCHS - PRETRAIN_EPOCHS - 1))
            lam = float(grl_lambda(progress))
            avg_cls_loss, avg_dom_loss = train_epoch_daan(
                feature_extractor, classifier,
                marginal_domain_clf, conditional_domain_clf,
                source_loader=train_loader, target_loader=target_loader,
                criterion_cls=criterion_cls, criterion_domain=criterion_domain,
                optimizer_main=optimizer_main, optimizer_domain=optimizer_domain,
                device=device,
                grl_lambda=lam,
                domain_loss_weight=DOMAIN_LOSS_WEIGHT
            )
            print(f"Epoch [{epoch+1}/{EPOCHS}] DAAN, cls_loss: {avg_cls_loss:.4f}, dom_loss: {avg_dom_loss:.4f}, grl_lambda: {lam:.4f}")

    # ========== 验证（QLD 的验证集） ==========
    y_true_val, y_prob_val = evaluate_model(feature_extractor, classifier, val_loader, device)
    y_pred_val = np.argmax(y_prob_val, axis=1)
    acc_qld = accuracy_score(y_true_val, y_pred_val)
    # 注意：确保 val 集有两个类别，否则 roc_auc_score 会报错
    try:
        auc_qld = roc_auc_score(y_true_val, y_prob_val[:, 1])
    except Exception:
        auc_qld = float('nan')
    cm_qld = confusion_matrix(y_true_val, y_pred_val)
    print(f"QLD 验证 Acc: {acc_qld:.4f}, AUC: {auc_qld if not np.isnan(auc_qld) else 'nan'}")
    print(classification_report(y_true_val, y_pred_val, digits=4))
    outer_metrics_qld.append({'acc': acc_qld, 'auc': auc_qld})
    outer_conf_qld.append(cm_qld)

    # ========== 外部测试集（WA） ==========
    X_test_tensor = X_vic_tensor_all
    y_test_tensor = y_vic_tensor_all
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    y_true_test, y_prob_test = evaluate_model(feature_extractor, classifier, test_loader, device)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    acc = accuracy_score(y_true_test, y_pred_test)
    try:
        auc = roc_auc_score(y_true_test, y_prob_test[:, 1])
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true_test, y_pred_test)
    print(f"WA 测试 Acc: {acc:.4f}, AUC: {auc if not np.isnan(auc) else 'nan'}")
    print(classification_report(y_true_test, y_pred_test, digits=4))
    outer_metrics_tests["WA"].append({'acc': acc, 'auc': auc})
    outer_conf_tests["WA"].append(cm)

    fold_num += 1

# =====================
# 7. 汇总结果
# =====================
print("\n=== QLD 总体 ===")
print("累加混淆矩阵:\n", sum(outer_conf_qld))
print("平均 Acc:", np.mean([m['acc'] for m in outer_metrics_qld]))
print("平均 AUC:", np.nanmean([m['auc'] for m in outer_metrics_qld]))

print(f"\n=== WA 总体 ===")
print("累加混淆矩阵:\n", sum(outer_conf_tests["WA"]))
print("平均 Acc:", np.mean([m['acc'] for m in outer_metrics_tests["WA"]]))
print("平均 AUC:", np.nanmean([m['auc'] for m in outer_metrics_tests["WA"]]))
