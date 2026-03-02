# dann_two_domain.py
import math
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


class DomainDiscriminator(nn.Module):
    """
    双域判别器：source=0, target=1
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # binary logits
        )

    def forward(self, f):
        return self.net(f)


class DANN(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int, num_classes: int):
        super().__init__()
        self.F = FeatureEncoder(input_channel=in_channels, feature_dim=feat_dim)
        self.C = LabelClassifier(feat_dim=feat_dim, num_classes=num_classes)
        self.D = DomainDiscriminator(feat_dim=feat_dim)

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
    p = step / max(1, total_steps)
    return float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


@torch.no_grad()
def eval_cls(model, loader, device):
    """
    评估分类准确率（NormalDataset: (x,y,...) 或 (x,y) 都行）
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    all_preds = []
    all_labels = []

    for batch in loader:
        # 兼容 (x,y) 或 (x,y,did)
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_logits, _, _ = model(x, alpha=0.0)
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


# ---------------------------
# Train (Two-domain DANN)
# ---------------------------
def train_dann_two_domain(
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

    model = DANN(
        in_channels=in_channels,
        feat_dim=feat_dim,
        num_classes=num_classes
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = min(len(source_loader), len(target_loader))
    total_steps = steps_per_epoch * epochs
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()

        it_s = iter(source_loader)
        it_t = iter(target_loader)

        cls_loss_avg = 0.0
        dom_loss_avg = 0.0
        n_batches = 0

        for _ in range(steps_per_epoch):
            batch_s = next(it_s)
            batch_t = next(it_t)

            # source: (x,y,*) 或 (x,y)
            if len(batch_s) == 2:
                x_s, y_s = batch_s
            else:
                x_s, y_s, _ = batch_s

            # target: (x,*) 或 (x,)
            # 兼容你现在的 TargetDataset 可能返回 (x,did)
            if isinstance(batch_t, (list, tuple)) and len(batch_t) >= 1:
                x_t = batch_t[0]
            else:
                x_t = batch_t

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            alpha = dann_alpha(global_step, total_steps)
            global_step += 1

            # 分类损失（source）
            y_logits_s, d_logits_s, _ = model(x_s, alpha=alpha)
            cls_loss = F.cross_entropy(y_logits_s, y_s)

            # 域损失（source=0, target=1）
            _, d_logits_t, _ = model(x_t, alpha=alpha)

            dom_y_s = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
            dom_y_t = torch.ones(x_t.size(0), dtype=torch.long, device=device)

            dom_loss_s = F.cross_entropy(d_logits_s, dom_y_s)
            dom_loss_t = F.cross_entropy(d_logits_t, dom_y_t)
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

        src_loss, src_acc,_,_ = eval_cls(model, source_loader, device)
        tgt_loss, tgt_acc,_,_ = eval_cls(model, val_loader, device)

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

    num_classes = config.DANN0_num_classes
    
    log_msg("=" * 80)
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 任务: {config.TASK},seed: {args.seed}")
    log_msg("=" * 80)
    log_msg(f"参数:")
    log_msg(f"  num_classes: {num_classes},batch_size: {config.DANN0_batch_size}, lr: {config.DANN0_lr}, weight_domain: {config.DANN0_weight_domain}, epochs: {config.DANN_epochs}")

    model = train_dann_two_domain(
        source_ds=source_ds,
        target_ds=target_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        in_channels=6,
        feat_dim=128,
        batch_size=config.DANN0_batch_size,
        lr=config.DANN0_lr,
        weight_domain=config.DANN0_weight_domain,
        epochs=config.DANN0_epochs,
        num_workers=8,
    )

    test_loader = DataLoader(
        test_ds, batch_size=config.DANN0_batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )
    log_msg(f"训练完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    avg_loss, accuracy, macro_f1, weighted_f1 = eval_cls(model, test_loader, device="cuda")
    log_msg(f"  准确率: {accuracy*100:.4f}% | Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    log_msg("=" * 80)
    with open(config.LOGS_DIR / 'DANN0_training.log', 'a') as f:
        f.write('\n'.join(log_messages) + '\n')
