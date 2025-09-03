import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from network.SolarSeerNet import SolarSeer  

# 自定义Dataset类，用于加载dummy数据
class SolarSeerDataset(Dataset):
    def __init__(self, satellite_data, clearghi_data, cloud_labels, irradiance_labels):
        self.satellite = torch.from_numpy(satellite_data).float()  # (N, 6, 4, 480, 1150)
        self.clearghi = torch.from_numpy(clearghi_data).float()    # (N, 24, 1, 480, 1150)
        self.cloud_labels = torch.from_numpy(cloud_labels).float() # (N, 24, 1, 480, 1150)
        self.irradiance_labels = torch.from_numpy(irradiance_labels).float()  # (N, 24, 1, 480, 1150)
    
    def __len__(self):
        return len(self.satellite)
    
    def __getitem__(self, idx):
        return self.satellite[idx], self.clearghi[idx], self.cloud_labels[idx], self.irradiance_labels[idx]

# 步骤4: 准备dummy数据集（维度基于论文：网格480x1150，卫星6小时x4通道；clearghi/cloud/irradiance 24小时x1通道）
# 训练集：100样本；验证集：20样本（模拟论文的训练/验证拆分）
num_train = 100
num_val = 20

# 卫星输入：过去6小时卫星观测，随机值模拟归一化图像（0-1）
satellite_train = np.random.rand(num_train, 6, 4, 480, 1150).astype(np.float32)
satellite_val = np.random.rand(num_val, 6, 4, 480, 1150).astype(np.float32)

# clear-sky GHI：未来24小时晴空辐照度，随机值模拟（0-1000 W/m²）
clearghi_train = np.random.rand(num_train, 24, 1, 480, 1150).astype(np.float32) * 1000
clearghi_val = np.random.rand(num_val, 24, 1, 480, 1150).astype(np.float32) * 1000

#  云覆盖标签：未来24小时云覆盖，随机值模拟（0-100%）
cloud_labels_train = np.random.rand(num_train, 24, 1, 480, 1150).astype(np.float32) * 100
cloud_labels_val = np.random.rand(num_val, 24, 1, 480, 1150).astype(np.float32) * 100

#  辐照度标签：未来24小时太阳辐照度，随机值模拟（0-1000 W/m²）
irradiance_labels_train = np.random.rand(num_train, 24, 1, 480, 1150).astype(np.float32) * 1000
irradiance_labels_val = np.random.rand(num_val, 24, 1, 480, 1150).astype(np.float32) * 1000

# 创建DataLoader
train_dataset = SolarSeerDataset(satellite_train, clearghi_train, cloud_labels_train, irradiance_labels_train)
val_dataset = SolarSeerDataset(satellite_val, clearghi_val, cloud_labels_val, irradiance_labels_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # batch_size小以模拟大网格内存需求
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 步骤5: 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SolarSeer().to(device)  # 实例化模型，内部包含cloud block和irradiance block

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 步骤6: 定义损失函数和优化器（基于论文评价指标RMSE/MAE，使用MSELoss）
criterion = nn.MSELoss()  # 均方误差，便于计算RMSE
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 假设学习率，实际可调
num_epochs = 10  # dummy训练用小epochs；实际论文训练一周，需更大

# 步骤7: 训练循环
for epoch in range(num_epochs):
    model.train()  # 训练模式
    train_loss = 0.0
    for satellite, clearghi, cloud_labels, irradiance_labels in train_loader:
        satellite, clearghi = satellite.to(device), clearghi.to(device)
        cloud_labels, irradiance_labels = cloud_labels.to(device), irradiance_labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播：假设模型forward接收卫星和clearghi，输出云覆盖和辐照度
        # (基于论文：cloud_block(satellite) -> cloud_forecast; irradiance_block(cloud_forecast + clearghi) -> irradiance_forecast)
        cloud_forecast, irradiance_forecast = model(satellite, clearghi)  # 需根据SolarSeerNet.py调整，如果forward不同
        
        # 计算损失：云覆盖损失 + 辐照度损失
        cloud_loss = criterion(cloud_forecast, cloud_labels)
        irradiance_loss = criterion(irradiance_forecast, irradiance_labels)
        loss = cloud_loss + irradiance_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证
    model.eval()  # 评估模式
    val_loss = 0.0
    with torch.no_grad():
        for satellite, clearghi, cloud_labels, irradiance_labels in val_loader:
            satellite, clearghi = satellite.to(device), clearghi.to(device)
            cloud_labels, irradiance_labels = cloud_labels.to(device), irradiance_labels.to(device)
            
            cloud_forecast, irradiance_forecast = model(satellite, clearghi)
            
            cloud_loss = criterion(cloud_forecast, cloud_labels)
            irradiance_loss = criterion(irradiance_forecast, irradiance_labels)
            loss = cloud_loss + irradiance_loss
            
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# 步骤8: 保存训练权重
torch.save(model.state_dict(), 'weight/solarseer_trained_weights.pth')
print("训练完成！权重已保存到 weight/solarseer_trained_weights.pth")
