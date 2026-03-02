from MEDGNet import Model
from MyNewDataset import NormalDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import os

class ClassifierNetwork(nn.Module):
    def __init__(self, in_dim, feat_dim=128, num_classes=10):
        super(ClassifierNetwork, self).__init__()
        self.head_fc1 = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.head_fc1(x)
        return x

def validate(model, classifier, data_loader, device, criterion, phase_name="Val"):
    """
    通用的评估函数 (用于验证集或测试集)
    """
    classifier.eval()
    
    loss_sum = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_data, y_data, d_data in data_loader:
            x_data, d_data = x_data.to(device), d_data.to(device)
            
            # 特征提取
            _, _,_,_, _,_,z,_=model(x_data,alpha = 0) 
            
            # 预测
            logits = classifier(z)
            loss = criterion(logits, d_data)
            
            loss_sum += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += d_data.size(0)
            correct += (predicted == d_data).sum().item()
            
    avg_loss = loss_sum / len(data_loader)
    acc = 100 * correct / total
    
    print(f"{phase_name} Results -> Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
    return avg_loss, acc

def main(epochs, num_classes, device, 
         train_x, train_y, train_info, train_domains, 
         val_x, val_y, val_info, val_domains,
         test_x, test_y, test_info, test_domains,
         pretrained_model_path="task4.pt",filter_classes=None):
    
    # --- 1. 数据加载 ---
    # Train
    train_ds = NormalDataset(x_path=train_x, y_path=train_y, info_path=train_info,
                             transform=None, filter_domains=train_domains,filter_classes=filter_classes, mmap_mode="r")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    
    # Validation
    val_ds = NormalDataset(x_path=val_x, y_path=val_y, info_path=val_info,
                           transform=None, filter_domains=val_domains,filter_classes=filter_classes,  mmap_mode="r")
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
    
    # Test (新增)
    test_ds = NormalDataset(x_path=test_x, y_path=test_y, info_path=test_info,
                            transform=None, filter_domains=test_domains,filter_classes=filter_classes,  mmap_mode="r")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    num_domains = len(train_ds.domain_to_id)
    
    # --- 2. 模型初始化 ---
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
    for param in model.parameters():
        param.requires_grad = False

    classifier = ClassifierNetwork(in_dim=128, feat_dim=128, num_classes=num_domains).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = None

    print("Start Training...")
    # --- 3. 训练循环 ---
    for epoch in range(1, epochs + 1):
        classifier.train()
        train_loss_sum = 0.0
        
        for x_data, y_data, d_data in train_loader:
            x_data, y_data, d_data = x_data.to(device), y_data.to(device), d_data.to(device)
            
            with torch.no_grad():
                #m = model.F(x_data)
                #_, z = model.C(m,alpha=1.0)
                _, _,_,_, _,_,z,_=model(x_data,alpha = 1.0) 
            
            logits = classifier(z)
            loss = criterion(logits, d_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        # 每个 Epoch 结束进行验证
        val_loss, val_acc = validate(model, classifier, val_loader, device, criterion, phase_name="Val")
        avg_train_loss = train_loss_sum / len(train_loader)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(classifier.state_dict())
            print(f"  *** New Best Val Model (Acc: {best_acc:.2f}%) ***")

    print("\nTraining Finished.")
    
    # --- 4. 加载最佳权重并进行测试 ---
    if best_model_wts is not None:
        print("Loading best model weights for testing...")
        classifier.load_state_dict(best_model_wts)
        
        # 保存一下最佳模型文件（可选）
        #torch.save(classifier.state_dict(), "best_domain_probe.pth")
        
        print("="*30)
        print("Final Evaluation on Test Set:")
        print("="*30)
        # 在测试集上运行，注意这里 phase_name="Test"
        validate(model, classifier, test_loader, device, criterion, phase_name="Test")

if __name__ == "__main__":
    main(
        epochs=50,
        num_classes=7,
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # 训练集参数
        train_x=r"/root/meta/dataset/train_x.npy", 
        train_y=r"/root/meta/dataset/train_y.npy",
        train_info=r"/root/meta/dataset/train_info.npy",
        train_domains=[(200,0),(400,0),(100,500),(300,500),(200,700),(400,700),(100,900),(300,900)],
        
        # 验证集参数
        val_x=r"/root/meta/dataset/val_x.npy",       
        val_y=r"/root/meta/dataset/val_y.npy",
        val_info=r"/root/meta/dataset/val_info.npy",
        val_domains=[(200,0),(400,0),(100,500),(300,500),(200,700),(400,700),(100,900),(300,900)],
        
        
        # 测试集参数
        test_x=r"/root/meta/dataset/test_x.npy",  
        test_y=r"/root/meta/dataset/test_y.npy",
        test_info=r"/root/meta/dataset/test_info.npy",
        test_domains=[(200,0),(400,0),(100,500),(300,500),(200,700),(400,700),(100,900),(300,900)],
        filter_classes=None
    )
