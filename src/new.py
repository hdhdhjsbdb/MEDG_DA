import torch
import torch.nn.functional as F
from MEDGNet import Model
from MyNewDataset import NormalDataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from MEDG import plot_tsne
import config
from datetime import datetime
import random
import argparse

def log_msg(msg):
    log_messages.append(msg)

@torch.no_grad()
def collect_z_d_y(model, loader, device="cuda"):
    model.eval()
    Z_list, D_list, Y_list,DOM_list = [], [], [],[]
    for x, y, dom in loader:
        x = x.to(device)
        y = y.to(device)
        dom = dom.to(device)

        y_logits, d_logits, _, m, z, d, rec = model(x, alpha=0.0)
        Z_list.append(z.detach().cpu())
        D_list.append(d.detach().cpu())
        Y_list.append(y.detach().cpu())
        DOM_list.append(dom.detach().cpu())

    Z = torch.cat(Z_list, dim=0)  # [N, 128]
    D = torch.cat(D_list, dim=0)  # [N, 128]
    Y = torch.cat(Y_list, dim=0)  # [N]
    DOM = torch.cat(DOM_list,dim=0)
    return Z, D, Y,DOM

def fit_proj_W(D, Z, ridge=1e-3):
    # D: [N, q], Z: [N, p]  (torch on CPU)
    Dm = D - D.mean(0, keepdim=True)
    Zm = Z - Z.mean(0, keepdim=True)

    q = Dm.size(1)
    I = torch.eye(q)

    # (D^T D + λI)^-1 D^T Z
    A = Dm.T @ Dm + ridge * I
    B = Dm.T @ Zm
    W = torch.linalg.solve(A, B)  # [q, p]
    return W, D.mean(0, keepdim=True), Z.mean(0, keepdim=True)

def make_z_clean(D, Z, W, D_mean, Z_mean):
    Dm = D - D_mean
    Zm = Z - Z_mean
    Z_hat = Dm @ W          # 可被 d 线性解释的部分（投影）
    Z_clean = Zm - Z_hat    # 去掉它
    return Z_clean, Z_hat


def train_linear_probe(X_train, y_train, X_test, y_test, num_classes, epochs=50, lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = nn.Linear(X_train.size(1), num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)

    X_train = X_train.to(device); y_train = y_train.to(device)
    X_test  = X_test.to(device);  y_test  = y_test.to(device)

    for epoch in range(epochs):
        clf.train()
        logits = clf(X_train)
        loss = F.cross_entropy(logits, y_train)
        print(f"{epoch}||loss:{loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        clf.eval()
    with torch.no_grad():
        pred = clf(X_test).argmax(1)
        acc = (pred == y_test).float().mean().item()
    return acc

if __name__ == "__main__":
    log_messages = []

    pretrained_model_path = config.pretrained_model_path
    num_classes = config.Domain_num_classes
    device = "cuda"

    log_msg("=" * 80)
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 模型地址: {pretrained_model_path}, 任务: {config.TASK},seed: {args.seed}")
    log_msg("=" * 80)

    # 验证集参数
    valid_x = config.DIRG_DATA_DIR / "val_x.npy"
    valid_y = config.DIRG_DATA_DIR / "val_y.npy"
    valid_info = config.DIRG_DATA_DIR / "val_info.npy"
    test_x = config.DIRG_DATA_DIR / "test_x.npy"
    test_y = config.DIRG_DATA_DIR / "test_y.npy"
    test_info = config.DIRG_DATA_DIR / "test_info.npy"

    src_domains = config.DIRG_task_src
    tgt_domains = config.DIRG_task_tgt

    val_ds = NormalDataset(x_path=val_x, y_path=val_y, info_path=val_info,
                           transform=None, filter_domains=tgt_domains,  mmap_mode="r")
    test_ds = NormalDataset(x_path=test_x, y_path=test_y, info_path=test_info,
                           transform=None, filter_domains=tgt_domains,  mmap_mode="r")
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds,batch_size=128,shuffle=False, num_workers=4)

    model = Model(in_channels=6, feat_dim=128, num_classes=num_classes, num_domains=12).to(device)
    if os.path.exists(pretrained_model_path):
        print(f"正在加载预训练模型权重: {pretrained_model_path}")
        # map_location 用于 CPU/GPU 兼容
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("预训练模型加载成功！")
    else:
        print(f"警告: 未找到预训练模型文件 {pretrained_model_path}，将使用随机初始化的模型（这可能毫无意义）。")
    model.eval()

    Zv, Dv, Yv,DOMv = collect_z_d_y(model, val_loader, device)
    Zt, Dt, Yt,DOMt = collect_z_d_y(model, test_loader, device)

    W, Zm, Dm = fit_proj_W(Zv, Dv, ridge=0.01)
    Dv_clean, _ = make_z_clean(Zv, Dv, W, Zm, Dm)
    Dt_clean, _ = make_z_clean(Zt, Dt, W, Zm, Dm)

    acc_D  = train_linear_probe(Dv, DOMv, Dt, DOMt, num_classes=8)
    acc_Dc = train_linear_probe(Dv_clean, DOMv, Dt, DOMt, num_classes=8)


    W, Dm, Zm = fit_proj_W(Dv, Zv, ridge=0.01)

    Zv_clean, _ = make_z_clean(Dv, Zv, W, Dm, Zm)
    Zt_clean, _ = make_z_clean(Dt, Zt, W, Dm, Zm)

    acc_Z  = train_linear_probe(Zv, Yv, Zt, Yt, num_classes=7)
    acc_Zc = train_linear_probe(Zv_clean, Yv, Zt, Yt, num_classes=7)


    Z_enhanced = 0.9 * Zt_clean + 0.1 * Dt_clean
    Zv_clean_concat = torch.cat([Zv_clean, Dv_clean], dim=1)
    Zt_clean_concat = torch.cat([Zt_clean, Dt_clean], dim=1) 
    acc_fin = train_linear_probe(Zv_clean_concat, Yv, Zt_clean_concat, Yt, num_classes=7)
    #plot_tsne(Zv_clean, Dv_clean, Yv,DOMv, save_path="tsne_output.pdf")

    log_msg(f"线性探针分类准确率:")
    log_msg(f"  原始 D: {acc_D:.4f}")
    log_msg(f"  去掉 D 线性可解释部分后的 D_clean: {acc_Dc:.4f}")
    log_msg(f"  原始 Z: {acc_Z:.4f}")
    log_msg(f"  去掉 D 线性可解释部分后的 Z_clean: {acc_Zc:.4f}")
    log_msg(f"  Z_clean + D_clean 线性拼接: {acc_fin:.4f}")

    log_mesg("=" * 80)
    with open(config.LOGS_DIR / 'Domains.log', 'a') as f:
        f.write('\n'.join(log_messages) + '\n') 

    print(acc_D)
    print(acc_Dc)
    print(acc_Z)
    print(acc_Zc)
    print(acc_fin)



