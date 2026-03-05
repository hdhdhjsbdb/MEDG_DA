import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_hierarchical_dataset(root_dir, save_dir, window_size=2048, stride=1024, expected_channels=8):
    """
    处理层级目录结构的数据集：
    Root/
      0/
        0g/
          600.csv
          ...
        6g/
        20g/
      1/
        ...
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_samples = []
    all_labels = []
    all_infos = []

    print(f"开始扫描目录: {root_dir}")

    # 1. 遍历类别文件夹 (0-6)
    # 使用 sorted 确保按顺序处理
    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        
        if not os.path.isdir(label_path):
            continue
        
        # 尝试将文件夹名转换为标签索引
        try:
            label = int(label_name)
        except ValueError:
            print(f"跳过非数字文件夹: {label_name}")
            continue

        # 2. 遍历负载文件夹 ("0g", "6g", "20g")
        for load_name in os.listdir(label_path):
            load_path = os.path.join(label_path, load_name)
            
            if not os.path.isdir(load_path):
                continue
            
            # 解析负载值 (去除'g'，转为int)
            try:
                load = int(load_name.replace('g', ''))
            except ValueError:
                print(f"跳过无法解析的负载文件夹: {load_name}")
                continue

            # 3. 遍历转速 CSV 文件
            for file_name in os.listdir(load_path):
                if not file_name.endswith('.csv'):
                    continue
                
                # 解析转速 (处理小数情况)
                try:
                    # 去掉后缀，先转为浮点数，再取整数部分
                    speed_part = os.path.splitext(file_name)[0]
                    speed = int(float(speed_part))  # 处理 "30.72" -> 30
                except ValueError:
                    print(f"跳过无法解析转速的文件: {file_name}")
                    continue

                file_full_path = os.path.join(load_path, file_name)

                try:
                    # 读取 CSV 文件
                    # 假设 CSV 没有表头。如果有表头，请改为 pd.read_csv(..., header=0)
                    df = pd.read_csv(file_full_path, header=None)
                    raw_signal = df.values # 转换为 numpy 数组

                    # 检查并统一形状为 (Channels, Length)
                    # CSV读取后通常是 -> (Length, Channels)
                    if raw_signal.ndim != 2:
                        print(f"文件 {file_name} 维度不对，跳过")
                        continue

                    # 如果列数等于通道数，说明是，需要转置为
                    if raw_signal.shape[1] == expected_channels:
                        raw_signal = raw_signal.T
                    # 如果行数等于通道数，说明已经是
                    elif raw_signal.shape[0] == expected_channels:
                        pass 
                    else:
                        print(f"文件 {file_name} 通道数不匹配 (期望 {expected_channels}, 实际 {raw_signal.shape})，跳过")
                        continue

                    # Z-Score 归一化 (按通道独立归一化)
                    mean = raw_signal.mean(axis=1, keepdims=True)
                    std = raw_signal.std(axis=1, keepdims=True)
                    raw_signal = (raw_signal - mean) / (std + 1e-8)

                    # 滑动窗口切片
                    L = raw_signal.shape[1]
                    # 如果数据长度小于窗口，跳过
                    if L < window_size:
                        continue
                        
                    for start in range(0, L - window_size + 1, stride):
                        chunk = raw_signal[:, start:start + window_size]
                        all_samples.append(chunk)
                        all_labels.append(label)
                        all_infos.append([speed, load])

                except Exception as e:
                    print(f"处理文件 {file_full_path} 时出错: {e}")
                    continue

    # 检查是否采集到数据
    if len(all_samples) == 0:
        print("错误：未采集到任何样本，请检查目录结构或文件格式！")
        return

    # 转换为数组
    X = np.array(all_samples, dtype=np.float32)
    Y = np.array(all_labels, dtype=np.int64)
    I = np.array(all_infos, dtype=np.int32)

    print(f"总计提取样本数: {len(X)}")
    print(f"标签分布: {np.unique(Y, return_counts=True)}")

    # 4. 全局分层划分 (保持与之前代码一致)
    # 70% 训练, 30% 临时
    X_train, X_temp, Y_train, Y_temp, I_train, I_temp = train_test_split(
        X, Y, I, test_size=0.3, random_state=42, stratify=Y
    )

    # 30% 临时 -> 15% 验证, 15% 测试
    X_val, X_test, Y_val, Y_test, I_val, I_test = train_test_split(
        X_temp, Y_temp, I_temp, test_size=0.5, random_state=42, stratify=Y_temp
    )

    # 5. 保存文件
    dataset_splits = {
        'train': (X_train, Y_train, I_train),
        'val': (X_val, Y_val, I_val),
        'test': (X_test, Y_test, I_test)
    }

    for split_name, (x, y, info) in dataset_splits.items():
        np.save(os.path.join(save_dir, f"{split_name}_x.npy"), x)
        np.save(os.path.join(save_dir, f"{split_name}_y.npy"), y)
        np.save(os.path.join(save_dir, f"{split_name}_info.npy"), info)
        print(f"{split_name} 集合保存成功: {len(x)} 个样本")


if __name__ == "__main__":
    # 修改为你的实际路径
    ROOT_DATA_PATH = r"E:\研一\实验\MEDG_DA\MAFDATA" # 修改为你的主目录路径
    SAVE_PATH = r"E:\研一\实验\MEDG_DA\data\MAFAULDA"
    
    WINDOW = 2048 
    STRIDE = 1024 # 这里的步长可以根据需要修改，如果设为2048则无重叠
    
    process_hierarchical_dataset(ROOT_DATA_PATH, SAVE_PATH, WINDOW, STRIDE, expected_channels=8)
