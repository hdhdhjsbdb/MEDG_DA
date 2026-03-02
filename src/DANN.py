# multi_domain_dann.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MEDGNet import FeatureEncoder
from MyNewDataset import NormalDataset, TargetDataset

import config
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='随机种子')
args = parser.parse_args()

# 如果指定了seed，就设置
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

def log_msg(msg):
    log_messages.append(msg)

# ---------------------------
# Gradient Reversal Layer
# ---------------------------
class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha: float):
    return GradReverseFn.apply(x, alpha)



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


class MultiDomainDiscriminator(nn.Module):
    """
    多域判别器：预测 K 个工况域 (domain_id)
    """
    def __init__(self, feat_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains)   # 多分类 logits
        )

    def forward(self, f):
        return self.net(f)  # [B, K]


class MultiDomainDANN(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int, num_classes: int, num_domains: int):
        super().__init__()
        self.F = FeatureEncoder()
        self.C = LabelClassifier(feat_dim=feat_dim, num_classes=num_classes)
        self.D = MultiDomainDiscriminator(feat_dim=feat_dim, num_domains=num_domains)

    def forward(self, x, alpha: float = 0.0):
        f = self.F(x)
        y_logits = self.C(f)
        f_rev = grad_reverse(f, alpha)
        d_logits = self.D(f_rev)
        return y_logits, d_logits, f


# ---------------------------
# Utils
# ---------------------------
def dann_alpha(step: int, total_steps: int) -> float:
    """
    DANN alpha schedule
    """
    p = step / max(1, total_steps)
    return float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


@torch.no_grad()
def eval_cls(model, loader, device):
    """
    评估分类准确率（NormalDataset: (x,y,d_domain)）
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    all_preds = []
    all_labels = []

    for x, y, _d in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_logits, _d_logits, _f = model(x, alpha=0.0)
        loss = F.cross_entropy(y_logits, y)
        pred = y_logits.argmax(dim=1)

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
        #micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    else:
        macro_f1 = weighted_f1 = 0.0

    return avg_loss, accuracy, macro_f1, weighted_f1


def build_global_domain_map(*datasets):
    """
    统一 source/target 的 domain_id 编号：0..K-1
    用 domains (speed,load) tuple 做键
    """
    all_domains = []
    for ds in datasets:
        all_domains.extend(ds.domains)

    # 去重并排序，保证可复现
    uniq = sorted(set(all_domains))
    global_domain_to_id = {dom: i for i, dom in enumerate(uniq)}
    return global_domain_to_id


# ---------------------------
# Train
# ---------------------------
def train_multi_domain_dann(
    source_ds,
    target_ds,
    val_ds,
    num_classes: int,
    in_channels: int = 6,
    feat_dim: int = 128,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_domain: float = 1.0,
    epochs: int = 50,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # DataLoaders
    source_loader = DataLoader(
        source_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # num_domains 必须来自全局 map 后的最大值
    num_domains = len(source_ds.domain_to_id)  # apply_global_map 后它就是 global map
    model = MultiDomainDANN(
        in_channels=in_channels,
        feat_dim=feat_dim,
        num_classes=num_classes,
        num_domains=num_domains
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = min(len(source_loader), len(target_loader))
    total_steps = steps_per_epoch * epochs
    global_step = 0
    bestacc=0.0

    for epoch in range(1, epochs + 1):
        model.train()

        it_s = iter(source_loader)
        it_t = iter(target_loader)

        cls_loss_avg = 0.0
        dom_loss_avg = 0.0
        n_batches = 0

        for _ in range(steps_per_epoch):
            # source: (x,y,did)
            x_s, y_s, did_s = next(it_s)
            # target: (x,did)
            x_t, did_t = next(it_t)

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            did_s = did_s.to(device, non_blocking=True)

            x_t = x_t.to(device, non_blocking=True)
            did_t = did_t.to(device, non_blocking=True)

            alpha = dann_alpha(global_step, total_steps)
            global_step += 1

            # 分类：只在 source
            y_logits_s, d_logits_s, _ = model(x_s, alpha=alpha)
            cls_loss = F.cross_entropy(y_logits_s, y_s)

            # 域判别：source + target 都用 “domain_id 多分类”
            # 注意：这里的 did_s/did_t 是 (speed,load) 的 global domain_id
            _, d_logits_t, _ = model(x_t, alpha=alpha)

            dom_loss_s = F.cross_entropy(d_logits_s, did_s)
            dom_loss_t = F.cross_entropy(d_logits_t, did_t)
            dom_loss = 0.5 * (dom_loss_s + dom_loss_t)

            loss = cls_loss + weight_domain * dom_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            cls_loss_avg += cls_loss.item()
            dom_loss_avg += dom_loss.item()
            n_batches += 1

        cls_loss_avg /= max(1, n_batches)
        dom_loss_avg /= max(1, n_batches)

        # 评估：source train acc + target-val acc
        src_loss, src_acc,_,_ = eval_cls(model, source_loader, device)
        tgt_loss, tgt_acc,_,_ = eval_cls(model, val_loader, device)
        if tgt_acc > bestacc:
            torch.save(model.state_dict(), "task1.pt")
            bestacc = tgt_acc
            
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_cls={cls_loss_avg:.4f} train_dom={dom_loss_avg:.4f} | "
            f"src_acc={src_acc*100:.2f}% src_loss={src_loss:.4f} | "
            f"tgt_val_acc={tgt_acc*100:.2f}% tgt_val_loss={tgt_loss:.4f}"
        )

    return model


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    log_messages = []
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
    # datasets
    source_ds = NormalDataset(
        x_path=train_x, y_path=train_y, info_path=train_info,
        transform=None, filter_domains=filter_domains_src, mmap_mode="r"
    )
    target_ds = TargetDataset(
        x_path=train_x, info_path=train_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    # 目标域验证集
    val_ds = NormalDataset(
        x_path=valid_x, y_path=valid_y, info_path=valid_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    test_ds = NormalDataset(
        x_path=test_x, y_path=test_y, info_path=test_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    num_classes = config.DANN_num_classes
    
    log_msg("=" * 80)
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 任务: {config.TASK},seed: {args.seed}")
    log_msg("=" * 80)
    log_msg(f"参数:")
    log_msg(f"  num_classes: {num_classes},batch_size: {config.DANN_batch_size}, lr: {config.DANN_lr}, weight_domain: {config.DANN_weight_domain}, epochs: {config.DANN_epochs}")

    # 统一全局ID
    global_map = build_global_domain_map(source_ds, target_ds, val_ds)
    source_ds.apply_global_map(global_map)
    target_ds.apply_global_map(global_map)
    val_ds.apply_global_map(global_map)
    test_ds.apply_global_map(global_map)
    model = train_multi_domain_dann(
        source_ds=source_ds,
        target_ds=target_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        in_channels=6,
        feat_dim=128,
        batch_size=config.DANN_batch_size,
        lr=config.DANN_lr,
        weight_domain=config.DANN_weight_domain,
        epochs=config.DANN_epochs,
        num_workers=8,
    )

    test_loader = DataLoader(
        test_ds, batch_size=config.DANN_batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )
    log_msg(f"训练完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    avg_loss, accuracy, macro_f1, weighted_f1 = eval_cls(model, test_loader, device="cuda")
    log_msg(f"  准确率: {accuracy*100:.4f}% | Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    log_msg("=" * 80)
    with open(config.LOGS_DIR / 'DANN_training.log', 'a') as f:
        f.write('\n'.join(log_messages) + '\n')



