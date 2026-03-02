import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import config
from MEDGNet import FeatureEncoder
from MyNewDataset import NormalDataset, TargetDataset
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='随机种子')
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

def log_msg(msg):
    log_messages.append(msg)


class LabelClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, f):
        return self.net(f)

class MCD_solver(object):
    def __init__(self, in_channels: int, feat_dim: int, num_classes: int, num_k: int):
        super().__init__()
        self.device = config.device
        self.F = FeatureEncoder().to(self.device)
        self.C1 = LabelClassifier(feat_dim=feat_dim, num_classes=num_classes).to(self.device)
        self.C2 = LabelClassifier(feat_dim=feat_dim, num_classes=num_classes).to(self.device)

        self.num_k = num_k  # 阶段C的迭代次数

        # 三个独立优化器
        self.opt_f = torch.optim.Adam(self.F.parameters(), lr=config.MCD_lr, weight_decay=0.0005)
        self.opt_c1 = torch.optim.Adam(self.C1.parameters(), lr=config.MCD_lr, weight_decay=0.0005)
        self.opt_c2 = torch.optim.Adam(self.C2.parameters(), lr=config.MCD_lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_f.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
    
    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))
    
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))
    
    def train(self, epoch, source_loader, target_loader, device):
        self.F.train()
        self.C1.train()
        self.C2.train()

        # ========== 阶段A: 源域预训练 ==========
        cls_loss_sum = 0.0
        n_batches = 0
        
        for x_s, y_s, did_s in source_loader:
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)

            f_s = self.F(x_s)
            output_s1 = self.C1(f_s)
            output_s2 = self.C2(f_s)

            cls_loss1 = F.cross_entropy(output_s1, y_s)
            cls_loss2 = F.cross_entropy(output_s2, y_s)
            cls_loss = cls_loss1 + cls_loss2

            cls_loss.backward()
            self.opt_f.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()
            
            cls_loss_sum += cls_loss.item()
            n_batches += 1
        
        avg_cls_loss = cls_loss_sum / max(1, n_batches)

        # ========== 阶段B: 最大化分类器差异 ==========
        iter_source = iter(source_loader)
        iter_target = iter(target_loader)

        for _ in range(min(len(source_loader), len(target_loader))):
            try:
                x_s, y_s, _ = next(iter_source)
                x_t, _ = next(iter_target)
            except StopIteration:
                break
            
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            # 源域：保持分类能力
            f_s = self.F(x_s)
            output_s1 = self.C1(f_s)
            output_s2 = self.C2(f_s)
            cls_loss1 = F.cross_entropy(output_s1, y_s)
            cls_loss2 = F.cross_entropy(output_s2, y_s)
            cls_loss = cls_loss1 + cls_loss2

            # 目标域：最大化差异
            f_t = self.F(x_t)
            output_t1 = self.C1(f_t)
            output_t2 = self.C2(f_t)
            loss_dis = self.discrepancy(output_t1, output_t2)

            loss = cls_loss - loss_dis
            loss.backward()  # 🔥 关键修复：添加backward！
            
            # 只更新分类器，不更新特征提取器
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

        # ========== 阶段C: 最小化分类器差异 ==========
        for k in range(self.num_k):
            for batch_idx, (x_t, _) in enumerate(target_loader):
                x_t = x_t.to(device, non_blocking=True)

                f_t = self.F(x_t)
                output_t1 = self.C1(f_t)
                output_t2 = self.C2(f_t)

                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()

                self.opt_f.step()
                self.reset_grad()
                
                # 每个num_k迭代只跑1个batch
                if batch_idx == 0:
                    break

        return avg_cls_loss


def eval_cls(solver, loader, device):
    solver.F.eval()
    solver.C1.eval()
    solver.C2.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            f = solver.F(x)
            out1 = solver.C1(f)
            out2 = solver.C2(f)

            # 集成预测：C1 + C2
            out_ensemble = out1 + out2
            loss = F.cross_entropy(out_ensemble, y)

            pred = out_ensemble.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.size(0)
    
    avg_loss = loss_sum / max(1, total)
    accuracy = correct / max(1, total)

    if len(all_labels) > 0:
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:
        macro_f1 = weighted_f1 = 0.0

    return avg_loss, accuracy, macro_f1, weighted_f1


def build_global_domain_map(*datasets):
    all_domains = []
    for ds in datasets:
        all_domains.extend(ds.domains)

    uniq = sorted(set(all_domains))
    global_domain_to_id = {dom: i for i, dom in enumerate(uniq)}
    return global_domain_to_id


def train_MCD(epochs, solver, source_loader, target_loader, val_loader, device):
    best_val_acc = 0.0
    best_epoch = 0
    best_F_state = None
    best_C1_state = None
    best_C2_state = None

    for epoch in range(1, epochs + 1):
        cls_loss = solver.train(epoch, source_loader, target_loader, device)
        tgtval_loss, tgtval_acc, _, _ = eval_cls(solver, val_loader, device)
        
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {cls_loss:.4f} | "
              f"Val Loss: {tgtval_loss:.4f} | "
              f"Val Acc: {tgtval_acc*100:.2f}%")

        if tgtval_acc > best_val_acc:
            best_val_acc = tgtval_acc
            best_epoch = epoch
            best_F_state = solver.F.state_dict()
            best_C1_state = solver.C1.state_dict()
            best_C2_state = solver.C2.state_dict()
            print(f"  >>> New best: Epoch {epoch}, Val Acc {tgtval_acc*100:.2f}%")
    
    # 加载最佳模型
    if best_F_state is not None:
        solver.F.load_state_dict(best_F_state)
        solver.C1.load_state_dict(best_C1_state)
        solver.C2.load_state_dict(best_C2_state)
        print(f"\nLoaded best model from epoch {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)")
    
    return solver


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    log_messages = []
    
    # 路径设置
    train_x = config.DIRG_DATA_DIR / "train_x.npy"
    train_y = config.DIRG_DATA_DIR / "train_y.npy"
    train_info = config.DIRG_DATA_DIR / "train_info.npy"

    valid_x = config.DIRG_DATA_DIR / "val_x.npy"
    valid_y = config.DIRG_DATA_DIR / "val_y.npy"
    valid_info = config.DIRG_DATA_DIR / "val_info.npy"

    test_x = config.DIRG_DATA_DIR / "test_x.npy"
    test_y = config.DIRG_DATA_DIR / "test_y.npy"
    test_info = config.DIRG_DATA_DIR / "test_info.npy"

    # source 训练域（有标签）
    filter_domains_src = config.DIRG_task_src
    # target 训练域（无标签）
    filter_domains_tgt = config.DIRG_task_tgt

    # 创建数据集
    source_ds = NormalDataset(
        x_path=train_x, y_path=train_y, info_path=train_info,
        transform=None, filter_domains=filter_domains_src, mmap_mode="r"
    )
    target_ds = TargetDataset(
        x_path=train_x, info_path=train_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )
    val_ds = NormalDataset(
        x_path=valid_x, y_path=valid_y, info_path=valid_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )
    test_ds = NormalDataset(
        x_path=test_x, y_path=test_y, info_path=test_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    # 全局域映射
    global_map = build_global_domain_map(source_ds, target_ds, val_ds)
    source_ds.apply_global_map(global_map)
    target_ds.apply_global_map(global_map)
    val_ds.apply_global_map(global_map)

    num_classes = config.MCD_num_classes

    # 日志
    log_msg("=" * 80)
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 任务: {config.TASK}, seed: {args.seed}")
    log_msg("=" * 80)
    log_msg(f"参数: num_classes={num_classes}, batch_size={config.MCD_batch_size}, lr={config.MCD_lr}, epochs={config.MCD_epochs}")

    # DataLoader（训练集shuffle=True）
    src_loader = DataLoader(
        source_ds, batch_size=config.MCD_batch_size, shuffle=True,  # 🔥 改为True
        num_workers=8, pin_memory=False, drop_last=True  # 🔥 改为0和False，避免多进程错误
    )

    tgt_loader = DataLoader(
        target_ds, batch_size=config.MCD_batch_size, shuffle=True,  # 🔥 改为True
        num_workers=8, pin_memory=False, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=config.MCD_batch_size, shuffle=False,
        num_workers=8, pin_memory=False, drop_last=False
    )

    # 初始化solver
    solver = MCD_solver(
        in_channels=6,
        feat_dim=128,
        num_classes=num_classes,
        num_k=4
    )

    # 训练
    solver = train_MCD(
        epochs=config.MCD_epochs,
        solver=solver,
        source_loader=src_loader,
        target_loader=tgt_loader,
        val_loader=val_loader,
        device=config.device
    )

    # 测试
    test_loader = DataLoader(
        test_ds, batch_size=config.MCD_batch_size, shuffle=False,
        num_workers=8, pin_memory=False, drop_last=False
    )
    
    log_msg(f"训练完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    avg_loss, accuracy, macro_f1, weighted_f1 = eval_cls(solver, test_loader, device=config.device)
    
    log_msg(f"测试结果: 准确率={accuracy*100:.4f}% | Loss={avg_loss:.4f} | Macro F1={macro_f1:.4f} | Weighted F1={weighted_f1:.4f}")
    log_msg("=" * 80)
    
    with open(config.LOGS_DIR / 'MCD_training.log', 'a') as f:
        f.write('\n'.join(log_messages) + '\n')