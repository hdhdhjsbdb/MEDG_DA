import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyNewDataset import ConDataset
import torch.nn.utils.spectral_norm as spectral_norm

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

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool1d(1)
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, w = x.size()
        x_w = self.pool_w(x)
        y = self.conv1(x_w)
        y = self.bn1(y)
        y = self.act(y)
        a_w = self.conv_w(y).sigmoid()
        return identity * a_w

# 2. 多尺度残差瓶颈块：纯 CNN 模拟 Transformer 的多频提取能力
class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        mid_ch = out_ch // 4
        # 1x1 降维
        self.conv_reduce = nn.Conv1d(in_ch, mid_ch * 4, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm1d(mid_ch * 4)
        
        # 三个尺度的并行路径
        # 路径1：短感受野 (捕捉尖峰)
        self.p1 = nn.Conv1d(mid_ch * 4, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        # 路径2：中感受野 (捕捉正常波动)
        self.p2 = nn.Conv1d(mid_ch * 4, mid_ch, kernel_size=15, stride=stride, padding=7, bias=False)
        # 路径3：长感受野 (空洞卷积，模拟全局视野)
        self.p3 = nn.Conv1d(mid_ch * 4, mid_ch * 2, kernel_size=3, stride=stride, 
                            padding=dilation, dilation=dilation, bias=False)
        
        self.bn_merge = nn.BatchNorm1d(out_ch)
        self.ca = CoordAtt(out_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn_reduce(self.conv_reduce(x)))
        
        out1 = self.p1(out)
        out2 = self.p2(out)
        out3 = self.p3(out)
        
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn_merge(out)
        out = self.ca(out) # 注入注意力
        
        out += identity
        return self.relu(out)

# 3. 高复杂度纯 CNN 特征提取器
class FeatureEncoder(nn.Module):
    def __init__(self, input_channel=6, feature_dim=128):
        super().__init__()
        
        # Stage 1: 宽卷积 Stem (WDCNN 思想)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channel, 64, kernel_size=64, stride=8, padding=28, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 2: 深层残差堆叠
        # 使用空洞卷积 (dilation) 逐层增加感受野
        self.layer1 = MultiScaleResidualBlock(64, 128, stride=2, dilation=1)
        self.layer2 = MultiScaleResidualBlock(128, 256, stride=2, dilation=2)
        self.layer3 = MultiScaleResidualBlock(256, 512, stride=2, dilation=4)
        self.layer4 = MultiScaleResidualBlock(512, 512, stride=1, dilation=8)
        
        # Stage 3: 全局特征聚合
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Stage 4: 投影头预处理 (Backbone 最终输出层)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        # Input: [B, 6, 2048]
        x = self.stem(x)    # [B, 64, 128]
        x = self.layer1(x)  # [B, 128, 64]
        x = self.layer2(x)  # [B, 256, 32]
        x = self.layer3(x)  # [B, 512, 16]
        x = self.layer4(x)  # [B, 512, 16]
        
        x = self.avgpool(x).flatten(1) # [B, 512]
        out = self.fc(x) # [B, feature_dim]
        return out

import torch.nn.utils.spectral_norm as spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            # 使用 Spectral Norm (谱归一化) 稳定对抗训练
            spectral_norm(nn.Linear(dim, dim)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(dim, dim)),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x)) # 残差连接

class StrongDiscriminator(nn.Module):
    def __init__(self, feat_dim, num_domains, hidden_dim=512):
        super().__init__()
        
        # 1. 映射层：迅速提升维度以捕捉细微特征
        self.input_layer = nn.Sequential(
            spectral_norm(nn.Linear(feat_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 2. 残差层：增加网络深度
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        # 3. 输出层
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_domains)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

class YourModel(nn.Module):
    def __init__(self, input_dim, num_classes, feature_dim=128):
        super(YourModel, self).__init__()
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # 特征头（用于正交损失）
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        # 提取基础特征
        base_features = self.feature_extractor(x)  # (batch_size, feature_dim)
        # 添加序列维度用于注意力
        d_seq = self.feature_head(base_features)  # (batch_size, feature_dim)
        x_seq = d_seq.unsqueeze(1)  # (batch_size, 1, feature_dim)
        # 自注意力
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        # 取第一个位置
        attended_features = attn_output.squeeze(1)  # (batch_size, feature_dim)
        # 最终分类
        logits = self.classifier(attended_features)
        return logits, base_features

class ReconDecoder(nn.Module):
    def __init__(self, feat_dim=128, dom_dim=128, out_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + dom_dim, hidden),  # 输入 feat + d
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)  # 输出重构任务特征
        )

    def forward(self, feat, z_dom):
        # 特征拼接
        x = torch.cat([feat, z_dom], dim=1)  # feat 和 d 连接
        return self.net(x)  # 返回重构的任务特征


class Model(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int, num_classes: int, num_domains: int):
        super().__init__()
        self.F = FeatureEncoder(input_channel=in_channels, feature_dim=128)
        self.C = YourModel(input_dim=128,feature_dim=128, num_classes=num_classes)
        self.D = StrongDiscriminator(feat_dim=128, num_domains=num_domains)
        self.DC = YourModel(input_dim=128,feature_dim=128, num_classes=num_domains)
        self.R = ReconDecoder(feat_dim=128, dom_dim=128, out_dim=128, hidden=256)

    def forward(self, x, alpha: float = 0.0):
        m = self.F(x)
        y_logits,z = self.C(m)
        dom,d = self.DC(m)
        feat_rev = grad_reverse(z, alpha)
        d_logits = self.D(feat_rev)
        rec = self.R(z, d)
        return y_logits, d_logits,dom, m,z,d,rec
