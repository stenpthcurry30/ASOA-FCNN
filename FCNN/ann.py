# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:35:26 2023

@author: DuMengLong
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import random
import math


# Set the random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Step 1: Load the data
data = pd.read_csv("to_ann2.csv")
#%%

data["ln_A"] = np.log(data["A_opt"])
data["Q_RT"] = data["Q_act_opt"] / (data["Temperature"] * 8.314)
data["ln_Strain_rate"] = np.log(data["Strain_rate"])

# Step 3: Preprocess the data
input_features = ["Strain", "ln_Strain_rate", "Temperature"]
output_features = ["alpha_opt","ln_A","n_opt","Q_RT"]
target_feature = "sigma_pred"

X = data[input_features].values
y = data[output_features].values
target = data[target_feature].values.reshape(-1, 1) 
#%%
# Normalize the input features using min-max normalization
X_min, X_max = X.min(axis=0), X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Standardize the output features using z-score normalization
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y = (y - y_mean) / y_std

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)


class MyDataset(Dataset):
    def __init__(self, X, y, target):
        self.X = X
        self.y = y
        self.target = target

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.target[idx]

dataset = MyDataset(X, y, target)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Shuffle and split data
indices = torch.randperm(len(dataset))

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size+val_size]
test_indices = indices[train_size+val_size:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)


batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#%% 设置模型
def xavier_initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        

class MaterialModel(nn.Module):
    def __init__(self, hidden_layers=2, hidden_nodes=40, activation='ReLU'):
        super(MaterialModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.activation = activation

        layers = []
        layers.append(nn.Linear(3, hidden_nodes))

        if self.activation == 'ReLU':
            activation_function = nn.ReLU()
        elif self.activation == 'LeakyReLU':
            activation_function = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        layers.append(activation_function)

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(activation_function)

        layers.append(nn.Linear(hidden_nodes, 4)) # Output 5 values (alpha, lnA, n, Q_RT, D)

        self.model = nn.Sequential(*layers)

        self.apply(xavier_initialize_weights)

    def forward(self, x):
        return self.model(x)

def constitutive_model(alpha, lnA, n, Q_RT, epsilon, ln_epsilon_dot, T):
    n = torch.where(n < 0, torch.ones_like(n) * 1e-8, n)
    T = torch.where(T < 0, torch.ones_like(T) * 1e-8, T)
    
    stress = (1.0 / alpha) * torch.asinh(torch.exp((ln_epsilon_dot - lnA + Q_RT) / n))
    return torch.unsqueeze(stress, dim=-1)


#%% 查看是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#%% 设置必要的函数
loss_fn = nn.MSELoss()



# 转换为torch tensor并且放到相应的设备上
y_mean = torch.tensor(y_mean, dtype=torch.float).to(device)
y_std = torch.tensor(y_std, dtype=torch.float).to(device)

X_min  = torch.tensor(X_min, dtype=torch.float).to(device)
X_max = torch.tensor(X_max, dtype=torch.float).to(device)
# learning_rate = 1e-4
# weight_decay = 1e-5

# optimizer = torch.optim.RMSprop(MaterialModel.parameters(), 
#                                   lr=learning_rate, 
#                                   weight_decay=weight_decay, 
#                                   momentum=0.9)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                          mode='min', 
#                                                          factor=0.1, 
#                                                          patience=10, 
#                                                          verbose=True)

# 定义一个函数，该函数依赖于当前的训练轮次
# def get_weight(epoch, total_epochs):
#     # 训练的前10%，权重为1
#     if epoch < total_epochs * 0.1:
#         return 1
#     # 训练的后10%，权重为0
#     elif epoch >= total_epochs * 0.9:
#         return 0
#     # 中间的80%，使用sigmoid函数进行平滑过渡
#     else:
#         # 将epoch归一化到[0,1]之间，然后再通过适当的偏移和缩放，使其在sigmoid函数中的输入范围为[-5,5]
#         normalized_epoch = 15 * ((epoch - total_epochs * 0.1) / (total_epochs * 0.9)) - 7.5
#         return 1 / (1 + np.exp(normalized_epoch))


def get_weight(epoch, total_epochs):
    if epoch < total_epochs * 0.3:
        return 1
    elif epoch >= total_epochs * 0.9:
        return 0
    else:
        normalized_epoch = math.pi * ((epoch - total_epochs * 0.3) / (total_epochs * 0.9 - total_epochs * 0.3))
        return 0.5 * (1 + math.cos(normalized_epoch))


# 设置训练代码
def train(dataloader, model, loss_fn, optimizer, epoch, total_epochs, device='cpu'):
    
    train_losses = []
    model.train()
    num_batches = len(dataloader)
    for batch, (X, y, target) in enumerate(dataloader):
        X, y, target = X.to(device), y.to(device), target.to(device)
        y_pred = model(X)
        optimizer.zero_grad()
        param_error = loss_fn(y_pred, y)
        
        # Clamp the outputs to be within the range of 0 and 1
        alpha_min = y[:, 0].min()  # the 0-th column corresponds to alpha_opt
        alpha_max = y[:, 0].max()
        alpha = torch.clamp(y_pred[:, 0], alpha_min, alpha_max)
        
        lnA_min = y[:, 1].min()  # the 1st column corresponds to ln_A
        lnA_max = y[:, 1].max()
        lnA = torch.clamp(y_pred[:, 1], lnA_min, lnA_max)
        
        n_min = y[:, 2].min()  # the 2nd column corresponds to n_opt
        n_max = y[:, 2].max()
        n = torch.clamp(y_pred[:, 2], n_min, n_max)
        
        Q_RT_min = y[:, 3].min()  # the 3rd column corresponds to Q_RT
        Q_RT_max = y[:, 3].max()
        Q_RT = torch.clamp(y_pred[:, 3], Q_RT_min, Q_RT_max)
        
        
        X_original = X * (X_max - X_min) + X_min
        
        # Reverse normalization of prediction
        alpha = alpha * y_std[0] + y_mean[0]
        lnA = lnA * y_std[1] + y_mean[1]
        n = n * y_std[2] + y_mean[2]
        Q_RT = Q_RT * y_std[3] + y_mean[3]
        
        # print(f"Shapes: alpha={alpha.shape}, lnA={lnA.shape}, n={n.shape}, Q_RT={Q_RT.shape}")

        # Compute the stress using the constitutive relation
        pred_stress = constitutive_model(alpha, lnA, n, Q_RT,  X_original[:, 0],  X_original[:, 1],  X_original[:, 2])
        
        # print(f"pred_stress shape: {pred_stress.float().shape}, target shape: {target.float().shape}")
        
        # Compute the errors for stress and material parameters
        stress_error = torch.sqrt(loss_fn(pred_stress.float(), target.float()))
        
        # print(f"target shape: {target.shape}")

        # Total loss is a weighted sum of stress and parameter errors
        weight = get_weight(epoch, total_epochs)
        loss = weight**2 * param_error + math.sqrt(1 - weight**2) * stress_error
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    # print(f"Avg Train loss: {avg_train_loss}")
    return avg_train_loss

def test(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y, target in dataloader:
            X, y, target = X.to(device), y.to(device), target.to(device)
            
            y_pred = model(X)
            
            # Clamp the outputs to be within the range of 0 and 1
            alpha_min = y[:, 0].min()  # the 0-th column corresponds to alpha_opt
            alpha_max = y[:, 0].max()
            alpha = torch.clamp(y_pred[:, 0], alpha_min, alpha_max)
            
            lnA_min = y[:, 1].min()  # the 1st column corresponds to ln_A
            lnA_max = y[:, 1].max()
            lnA = torch.clamp(y_pred[:, 1], lnA_min, lnA_max)
            
            n_min = y[:, 2].min()  # the 2nd column corresponds to n_opt
            n_max = y[:, 2].max()
            n = torch.clamp(y_pred[:, 2], n_min, n_max)
            
            Q_RT_min = y[:, 3].min()  # the 3rd column corresponds to Q_RT
            Q_RT_max = y[:, 3].max()
            Q_RT = torch.clamp(y_pred[:, 3], Q_RT_min, Q_RT_max)
            
    
            
            # Reverse normalization of prediction
            alpha = alpha * y_std[0] + y_mean[0]
            lnA = lnA * y_std[1] + y_mean[1]
            n = n * y_std[2] + y_mean[2]
            Q_RT = Q_RT * y_std[3] + y_mean[3]
            
            X_original = X * (X_max - X_min) + X_min
            
            # Compute the stress using the constitutive relation
            pred_stress = constitutive_model(alpha, lnA, n, Q_RT, X_original[:, 0], X_original[:, 1], X_original[:, 2])
            
            test_loss += torch.sqrt(loss_fn(pred_stress.float(), target.float())).item()
    test_loss /= size
    print(f"Test loss: {test_loss}")
    return test_loss

def train_model(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, scheduler, loss_fn, device, epochs, early_stopping_patience, model_weights_path):
    best_valid_loss = float('inf')
    epochs_without_improvement = 0

    train_losses, valid_losses, test_losses = [], [], []

    for epoch in range(epochs):
        print(f"\n{'-'*20} Epoch {epoch+1} {'-'*20}")

        # 训练阶段
        train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch, epochs, device)
        train_losses.append(train_loss)
        
        # 打印训练误差
        print(f"Train loss: {train_loss:.4f}")

        # 验证阶段
        valid_loss = test(valid_dataloader, model, loss_fn, device)
        valid_losses.append(valid_loss)

        # 计算测试损失
        test_loss = test(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)

        # 更新学习率调度器
        scheduler.step()

        # 检查验证损失是否有所改善，如果是，则保存模型权重
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_weights_path)
            print(f"New best validation loss: {valid_loss:.4f}, saving model weights to {model_weights_path}")
        else:
            epochs_without_improvement += 1
            print(f"Validation loss: {valid_loss:.4f}")

        # 早停
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\n{'-'*20} Final Results {'-'*20}")
    print(f"Final Train loss: {train_loss}")
    print(f"Final Validation loss: {valid_loss}")
    print(f"Final Test loss: {test_loss}")
    
    return train_losses, valid_losses, test_losses

#%% 自动寻优
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader
import math

class DecayAmplitudeCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, amplitude_decay=1.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.amplitude_decay = amplitude_decay
        super(DecayAmplitudeCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr * self.amplitude_decay ** (self.last_epoch // self.T_max) - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch % self.T_max) / self.T_max)) / 2
                for base_lr in self.base_lrs]

    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                          self.T_max, 
                                                          self.last_epoch)

    def step(self, epoch=None):
        if epoch is not None and epoch != 0 and (epoch % self.T_max == 0):
            self.base_lrs = [base_lr * self.amplitude_decay for base_lr in self.base_lrs]
        return super(DecayAmplitudeCosineAnnealingLR, self).step(epoch)

best_valid_loss = float("inf")
best_trial_weights_path = "best_model_weights_of_material_para.pth"
#%%
import torch
from torch.utils.data import DataLoader

best_params = {
    "hidden_layers": 27,
    "hidden_nodes": 59,
    "learning_rate": 3.5087698290849404e-05,
    "weight_decay": 9.173330278904162e-10,
    "batch_size": 64,
    "activation": "ReLU",
    "optimizer": "SGD",
    "eta_min": 7.583278180713205e-10,
    "T_max": 740,  # 注意这里是74*10
    "amplitude_decay": 0.2326214473845283
}

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = MaterialModel(
    hidden_layers=best_params["hidden_layers"],
    hidden_nodes=best_params["hidden_nodes"],
    activation=best_params["activation"]
).to(device)

# 设置优化器
if best_params["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=best_params["learning_rate"], 
        weight_decay=best_params["weight_decay"], 
        momentum=0.9
    )

# 设置学习率调度器
scheduler = DecayAmplitudeCosineAnnealingLR(
    optimizer,
    T_max=best_params["T_max"],
    eta_min=best_params["eta_min"],
    amplitude_decay=best_params["amplitude_decay"]
)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

# 训练参数
n_epochs = 3000
early_stopping_patience = 500

# 训练模型
train_losses, valid_losses, test_losses = train_model(
    model=model,
    train_dataloader=train_loader,
    valid_dataloader=val_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,  
    device=device,
    epochs=n_epochs,
    early_stopping_patience=early_stopping_patience,
    model_weights_path="best_model_weights.pth"  
)

best_valid_loss = min(valid_losses)
torch.save(model.state_dict(), "best_model_weights.pth")

import pickle

loss_history = {
    "train_losses": train_losses,
    "valid_losses": valid_losses,
    "test_losses": test_losses
}

with open("loss_history.pkl", "wb") as file:
    pickle.dump(loss_history, file)














