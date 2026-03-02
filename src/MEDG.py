import os
import torch
from torch.func import functional_call
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MEDGNet import Model
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from typing import List, Tuple
import math
from MyNewDataset import NormalDataset, TargetDataset
import config



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

def named_params_dict(module):
    """获取模型所有参数的字典"""
    return {k: v for k, v in module.named_parameters()}

def load_pretrained_encoder(model, pretrained_path, freeze=False):
    """
    加载 DomainAwareMoco 预训练的 encoder_q 到 model.F
    """
    if not os.path.exists(pretrained_path):
        print(f"Pretrained path not found: {pretrained_path}")
        return
    
    print(f"Loading pretrained encoder from {pretrained_path}...")
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    # 直接加载到 F（FeatureEncoder）
    missing, unexpected = model.F.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    # 冻结选项
    if freeze:
        for param in model.F.parameters():
            param.requires_grad = False
        print("Encoder frozen!")
    
    print("Pretrained encoder loaded successfully!")
# ---------------------------
# 元学习前向传播函数
# ---------------------------
def meta_fwd_cls(params, model, x, y, alpha):
    """
    专门用于元学习的前向传播函数（只计算分类损失）
    params: 模型的参数字典
    """
    # 剥离前缀以匹配 model 内部的成员变量名
    # 假设你的 Model 类里 FeatureEncoder 叫 F, Classifier 叫 C
    model_p = {k.replace("model.", ""): v for k, v in params.items() if k.startswith("model.")}
    
    # 使用 functional_call 进行前向计算
    # 注意：这里只取 y_logits
    y_logits,_, _, _, _,_,_= functional_call(model, model_p, (x, alpha))
    return F.cross_entropy(y_logits, y)

# ---------------------------
# 全局域映射函数
# ---------------------------
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
# CORAL 损失函数
# ---------------------------
class CoralLoss(nn.Module):
    """
    CORAL Loss 用于对齐源域和目标域特征的二阶统计量（协方差）
    """
    def forward(self, source, target):
        d = source.size(1)
        # 源域协方差
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = torch.matmul(xm.t(), xm) / (source.size(0) - 1)
        # 目标域协方差
        xmt = torch.mean(target, 0, keepdim=True) - target
        xtc = torch.matmul(xmt.t(), xmt) / (target.size(0) - 1)
        # 弗罗贝尼乌斯范数
        loss = torch.norm(xc - xtc, p='fro')
        return loss / (4 * d * d)

# ---------------------------
# HSIC 损失函数
# ---------------------------
def hsic_loss1(x, y):
    """
    自适应带宽的 HSIC 实现
    """
    B = x.size(0)
    if B < 2: return torch.tensor(0.0).to(x.device)

    def d_matrix(z):
        # 计算平方欧式距离矩阵
        dist = torch.cdist(z, z) ** 2
        return dist

    def rbf_kernel(dist_matrix):
        # Median Trick: 使用距离的中位数作为 sigma^2
        sigma2 = torch.median(dist_matrix[dist_matrix > 0])
        # 防止除以0
        if sigma2 == 0: sigma2 = 1.0
        return torch.exp(-dist_matrix / (2 * sigma2))

    dist_x = d_matrix(x)
    dist_y = d_matrix(y)
    
    K = rbf_kernel(dist_x)
    L = rbf_kernel(dist_y)

    H = torch.eye(B, device=x.device) - (1.0 / B) * torch.ones((B, B), device=x.device)
    
    # HSIC 计算
    # Trace(KHKL) / (B-1)^2
    KH = H @ K @ H
    LH = H @ L @ H
    
    return torch.trace(KH @ LH) / ((B - 1) ** 2)
def hsic_loss(x, y, sigma=1.0):
    """
    x, y: [B, D]
    """
    B = x.size(0)

    def rbf_kernel(z):
        dist = torch.cdist(z, z) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    K = rbf_kernel(x)
    L = rbf_kernel(y)

    H = torch.eye(B, device=x.device) - (1.0 / B) * torch.ones((B, B), device=x.device)
    KH = H @ K @ H
    LH = H @ L @ H

    return torch.trace(KH @ LH) / ((B - 1) ** 2)

# ---------------------------
# 训练函数
# ---------------------------
def train(
    source_ds, target_ds, val_ds,
    num_classes: int,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_domain: float = 1,  
    weight_coral: float = 0.5,
    weight_domainacc: float = 0.7,
    weight_outer = 0.7,
    weight_HSIC = 0.5,
    weight_rec = 0.1,
    device: str = "cuda",
    save_name: str = "task4"
):
    source_loader = DataLoader(source_ds, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    num_domains = len(source_ds.domain_to_id)
    model = Model(in_channels=6, feat_dim=128, num_classes=num_classes, num_domains=num_domains).to(device)
    #load_pretrained_encoder(model, "con_vis/weights/encoder_q_best.pth", freeze=False)
    #print("FeatureExtractor parameters:", sum(p.numel() for p in model.F.parameters()))

    optimizer = torch.optim.Adam([
        {"params": model.F.parameters(),  "lr": 0.0005},  # encoder 小一点
        {"params": model.C.parameters(),  "lr": 0.0005},  # 故障头
        {"params": model.DC.parameters(), "lr": 0.0005},  # 域特征头
        {"params": model.D.parameters(),  "lr": 0.0005},  # 域判别器
        {"params": model.R.parameters(),  "lr": 0.0005},  # 重构头
    ], weight_decay=1e-4)
    criterion_coral = CoralLoss()

    steps_per_epoch = min(len(source_loader), len(target_loader))
    best_acc = 0.0 

    for epoch in range(1, epochs + 1):
        model.train()
        it_s, it_t = iter(source_loader), iter(target_loader)
        
        for step in range(steps_per_epoch):
            x_s, y_s, did_s = next(it_s)
            x_t, did_t = next(it_t)
            if step == 0 and epoch == 1:
                print(did_s)
                print(did_t)
            
            x_s, y_s, did_s = x_s.to(device), y_s.to(device), did_s.to(device)
            x_t, did_t = x_t.to(device), did_t.to(device)

            # 计算 Alpha (对抗系数)
            p = (step + (epoch-1)*steps_per_epoch) / (epochs * steps_per_epoch)
            alpha = 2. / (1. + math.exp(-10 * p)) - 1
            '''
            # ===== 1. 构建元学习任务 (Meta-Task Construction) =====
            # 按域划分源域数据
            unique_domains = torch.unique(did_s).tolist()
            
            if len(unique_domains) < 2:
                # 如果域的数量少于2，则进行普通的训练集/测试集切分
                split_idx = x_s.size(0) // 2
                x_tr, y_tr, did_tr = x_s[:split_idx], y_s[:split_idx], did_s[:split_idx]
                x_te, y_te, did_te = x_s[split_idx:], y_s[split_idx:], did_s[split_idx:]
            else:
                # 随机选择n个域作为元测试集 (meta-test)
                test_domains = random.sample(unique_domains, 2)
                
                # Ensure test_domains is on the same device as did_s
                test_domains = torch.tensor(test_domains).to(did_s.device)
                
                # 创建掩码
                test_mask = torch.isin(did_s, test_domains)
                train_mask = ~test_mask  # 其余所有域作为训练集
                
                # 执行切分
                x_tr, y_tr, did_tr = x_s[train_mask], y_s[train_mask], did_s[train_mask]
                x_te, y_te, did_te = x_s[test_mask], y_s[test_mask], did_s[test_mask]
            # ===== 2. 元学习内环 (Inner Loop) =====
            # 备份当前模型参数（用于元学习）
            current_params = {f"model.{k}": v for k, v in named_params_dict(model).items()}
            loss_inner = meta_fwd_cls(current_params, model, x_tr, y_tr, alpha)
            grads = torch.autograd.grad(
                loss_inner, current_params.values(), 
                create_graph=True, allow_unused=True
            )
            # 手动更新得到 fast_params
            fast_params = {}
            for (name, p), g in zip(current_params.items(), grads):
                if g is not None:
                    fast_params[name] = p - 0.0001 * g
                else:
                    fast_params[name] = p
             # ===== 3. 元学习外环 (Outer Loop) =====
            loss_outer = meta_fwd_cls(fast_params, model, x_te, y_te, alpha)
            '''

            # ===== 4. 计算其他损失并总合 =====
            # 前向计算
            y_logits, d_s_logits,dom_s, m_s,z_s,d_s,rec_s = model(x_s, alpha=alpha)
            _, d_t_logits,dom_t, m_t,z_t,d_t,rec_t = model(x_t, alpha=alpha)

            loss_normal = F.cross_entropy(y_logits, y_s)

            # 1. 域对抗损失 (多域辨别)
            loss_dom = 0.5 * (F.cross_entropy(d_s_logits, did_s) + F.cross_entropy(d_t_logits, did_t))

            # 2. CORAL 对齐损失 
            loss_coral = criterion_coral(m_s, m_t)

            # 3. 域分类准确率损失
            dom_loss = 0.5*(F.cross_entropy(dom_s, did_s)+F.cross_entropy(dom_t, did_t))
            # 4. 正交损失
            orth_loss = 0.5*(hsic_loss1(z_s, d_s)+hsic_loss1(z_t, d_t))
            # 5. 重构损失
            rec_loss = 0.5*(F.mse_loss(rec_s, z_s.detach())+ F.mse_loss(rec_t, z_t.detach()))

            # 总损失
            #total_loss = loss_inner + weight_outer*loss_outer + weight_domain * loss_dom + weight_coral * loss_coral + weight_domainacc * dom_loss + weight_HSIC*orth_loss+weight_rec*rec_loss
            total_loss = loss_normal + weight_domain * loss_dom + weight_coral * loss_coral + weight_domainacc * dom_loss + weight_HSIC*orth_loss+weight_rec*rec_loss


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 验证逻辑 (保持不变)
        _, src_acc,src_dom = eval_cls(model,source_loader, device)
        _, tgt_acc,tgt_dom = eval_cls(model, val_loader, device)
        print(f"Epoch {epoch:03d} | "
            f"Coral: {loss_coral.item():.4f} | "
            f"HSIC LOSS: {orth_loss.item():.4f} | "
            f"adv LOSS: {loss_dom.item():.4f} | "
            f"Src Acc: {src_acc*100:.2f}% | "
            f"Src Dom: {src_dom*100:.2f}% | "
            f"Tgt Acc: {tgt_acc*100:.2f}% | "
            f"Tgt Dom: {tgt_dom*100:.2f}%")
        if tgt_acc > best_acc:
            best_acc = tgt_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'global_map': global_map,  # 非常重要：确保推理时的 Domain ID 一致
        }
            save_path = f"{save_name}.pt"
            torch.save(state, save_path)
            print(f" >>> Best model with Tgt_Acc: {best_acc*100:.2f}%")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n✓ Loaded best model with Tgt_Acc: {best_acc*100:.2f}%")
    else:
        print("Warning: No best model found during training!")
    
    return model  # 现在返回的是验证集上表现最好的模型

@torch.no_grad()
def eval_cls1(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_dom, correct_dom = 0, 0
    loss_sum = 0
    all_z, all_d, all_labels,all_domains = [], [], [] ,[]# 用于存储 z 特征, d 特征和标签

    for x, y, d in loader:
        x, y, d = x.to(device), y.to(device), d.to(device)
        logits, _, logits_dom, _, _, _, _ = model(x, alpha=0.0)
        loss = F.cross_entropy(logits, y)

        # 计算分类准确率
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()

        # 计算域分类准确率
        pred_dom = logits_dom.argmax(1)
        correct_dom += (pred_dom == d).sum().item()

        total += y.size(0)
        total_dom += d.size(0)
        loss_sum += loss.item() * y.size(0)

        # 保存特征 z 和 d 以及标签 y
        all_z.append(logits.detach().cpu().numpy())  # 存储 z 特征
        all_d.append(logits_dom.detach().cpu().numpy())  # 存储 d 特征
        all_labels.append(y.detach().cpu().numpy())  # 存储标签 y
        all_domains.append(d.detach().cpu().numpy())

    # 合并所有特征
    all_z = np.concatenate(all_z, axis=0)
    all_d = np.concatenate(all_d, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)

    return loss_sum / total, correct / total, correct_dom / total_dom, all_z, all_d, all_labels,all_domains
# ---------------------------
# 评估函数
# ---------------------------
@torch.no_grad()
def eval_cls(model,loader, device):
    model.eval()
    total, correct = 0, 0
    total_dom,correct_dom = 0,0
    loss_sum = 0
    for x, y, d in loader:
        x, y ,d= x.to(device), y.to(device), d.to(device)
        logits,_,logits_dom,_,_,_,_ = model(x, alpha=0.0)
        loss = F.cross_entropy(logits, y)

        # 计算分类准确率
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()

        # 计算域分类准确率
        pred_dom = logits_dom.argmax(1)
        correct_dom += (pred_dom == d).sum().item()

        total += y.size(0)
        total_dom += d.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total, correct_dom/total_dom

def test(model,target_ds, batch_size=64, device='cuda', save_path='test_results.pdf'):
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=False)
    loss, acc, dom_acc, all_z, all_d, all_labels,all_domain_labels = eval_cls1(model, target_loader, device)
    # 输出结果
    print(f"Test Loss: {loss:.4f}, Target Domain Accuracy: {acc * 100:.2f}%")
    print(f"Domain Classification Accuracy: {dom_acc * 100:.2f}%")
    # 使用 t-SNE 绘制 z 和 d 特征
    plot_tsne(all_z, all_d, labels=all_labels,domain_labels=all_domain_labels, save_path=save_path)
    print(f"t-SNE plots saved to {save_path}")
def plot_tsne(z_features, d_features, labels,domain_labels, save_path="tsne_output.pdf"):
    # 使用 t-SNE 将特征降维到 2D
    tsne = TSNE(n_components=2, random_state=42)
    z_embedded = tsne.fit_transform(z_features)
    d_embedded = tsne.fit_transform(d_features)

    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    # 绘制 z 特征与故障的 t-SNE 图
    axs[0].scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels, cmap='jet', s=10)
    axs[0].set_title('t-SNE of z Features (with Faults)')
    axs[0].set_xlabel('t-SNE Component 1')
    axs[0].set_ylabel('t-SNE Component 2')

    # 绘制 z 特征与域的 t-SNE 图
    axs[1].scatter(z_embedded[:, 0], z_embedded[:, 1], c=domain_labels, cmap='jet', s=10)
    axs[1].set_title('t-SNE of z Features (with Domains)')
    axs[1].set_xlabel('t-SNE Component 1')
    axs[1].set_ylabel('t-SNE Component 2')

    # 绘制 d 特征与域的 t-SNE 图
    axs[2].scatter(d_embedded[:, 0], d_embedded[:, 1], c=domain_labels, cmap='jet', s=10)
    axs[2].set_title('t-SNE of d Features (with Domains)')
    axs[2].set_xlabel('t-SNE Component 1')
    axs[2].set_ylabel('t-SNE Component 2')

    # 保存为 PDF 文件
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
# ---------------------------
# Main (执行入口)
# ---------------------------
if __name__ == "__main__":
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
    filter_domains_src = config.DIRG_task4_src
    # target 训练域（无标签）
    filter_domains_tgt = config.DIRG_task4_tgt
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
    

    # 统一全局ID
    global_map = build_global_domain_map(source_ds, target_ds, val_ds)
    source_ds.apply_global_map(global_map)
    target_ds.apply_global_map(global_map)
    val_ds.apply_global_map(global_map)
    test_ds.apply_global_map(global_map)

    model = train(
        source_ds=source_ds, 
        target_ds=target_ds, 
        val_ds=val_ds, 
        num_classes=config.num_classes,
        epochs = config.epochs,
        weight_coral=config.weight_coral,
        weight_domain = config.weight_adv,
        weight_domainacc = config.weight_domainacc,
        weight_outer = config.weight_outer,
        weight_HSIC = config.weight_HSIC,
        weight_rec = config.weight_rec,
        save_name = "task4"         
    )
    test(model,test_ds,64)
