# File: net.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def load_drosophila_matrix(csv_path, apply_pruning=False, signed=False):
    """加载并处理果蝇连接矩阵"""
    W_df = pd.read_csv(csv_path, index_col=0, header=0)
    W = W_df.values.astype(np.float32)
  
    # 根据 signed 参数选择归一化方式
    if signed:
        max_abs = np.max(np.abs(W))
        W_norm = W / max_abs if max_abs != 0 else W
    else:
        W_min, W_max = W.min(), W.max()
        W_norm = (W - W_min) / (W_max - W_min + 1e-8)
  
    if apply_pruning:
        non_zero_count = np.count_nonzero(W)
        return W_norm, non_zero_count
    return W_norm

def load_connectivity_data(connectivity_path, annotation_path):
    """加载并预处理连接矩阵和注释数据"""
    # 加载注释文件
    df_annot = pd.read_csv(annotation_path)
  
    # 提取视觉感觉神经元ID
    mask = (df_annot['celltype'] == 'sensory') & (df_annot['additional_annotations'] == 'visual')
    sensory_visual_ids = []
    for _, row in df_annot[mask].iterrows():
        for col in ['left_id', 'right_id']:
            if (id_str := str(row[col]).lower()) != "no pair":
                sensory_visual_ids.append(int(id_str))
  
    # 去重排序
    sensory_visual_ids = sorted(list(set(sensory_visual_ids)))
    print(f"Found {len(sensory_visual_ids)} sensory-visual neuron IDs")
  
    # 加载连接矩阵
    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
  
    # 筛选有效ID
    valid_sensory_ids = [nid for nid in sensory_visual_ids if nid in df_conn.index]
    other_ids = [nid for nid in df_conn.index if nid not in valid_sensory_ids]
  
    # 重新排序矩阵
    df_reindexed = df_conn.loc[valid_sensory_ids + other_ids, valid_sensory_ids + other_ids]
  
    # 标准化并拆分矩阵
    adj_matrix = df_reindexed.values * 1e-3  # 统一标准化
  
    num_S = len(valid_sensory_ids)
    return {
        'W_ss': adj_matrix[:num_S, :num_S],
        'W_sr': adj_matrix[:num_S, num_S:],
        'W_rs': adj_matrix[num_S:, :num_S],
        'W_rr': adj_matrix[num_S:, num_S:],
        'sensory_ids': valid_sensory_ids
    }

class DrosophilaRNN(nn.Module):
    def __init__(self, input_dim, sensory_dim, residual_dim, num_classes, conn_weights):
        super().__init__()
        self.W_ss = nn.Parameter(torch.tensor(conn_weights['W_ss'], dtype=torch.float32), requires_grad=True)
        self.W_sr = nn.Parameter(torch.tensor(conn_weights['W_sr'], dtype=torch.float32), requires_grad=True)
        self.W_rs = nn.Parameter(torch.tensor(conn_weights['W_rs'], dtype=torch.float32), requires_grad=True)
        self.W_rr = nn.Parameter(torch.tensor(conn_weights['W_rr'], dtype=torch.float32), requires_grad=True)
      
        self.input_proj = nn.Linear(input_dim, sensory_dim)
        self.output_layer = nn.Linear(residual_dim, num_classes)
        self.activation = nn.ReLU()

        assert self.W_ss.shape == (sensory_dim, sensory_dim)
        assert self.W_sr.shape == (sensory_dim, residual_dim)
        assert self.W_rs.shape == (residual_dim, sensory_dim)
        assert self.W_rr.shape == (residual_dim, residual_dim)

    def forward(self, x, time_steps=10):
        batch_size = x.shape[0]
        device = x.device
      
        S = torch.zeros(batch_size, self.W_ss.shape[0], device=device)
        R = torch.zeros(batch_size, self.W_rr.shape[0], device=device)
      
        E = self.input_proj(x)  # [batch_size, sensory_dim]
      
        for t in range(time_steps):
            E_t = E if t % 5 == 0 else torch.zeros_like(E)
          
            S_next = self.activation(
                S @ self.W_ss +    # S->S连接
                E_t +             # 外部输入
                R @ self.W_rs     # R->S连接
            )
            R_next = self.activation(
                R @ self.W_rr +    # R->R连接
                S @ self.W_sr      # S->R连接
            )
          
            S, R = S_next, R_next
      
        return self.output_layer(R)
    
class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 W_init=None, W_ref=None, trainable=True, pruning_method=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=True)
        self.pruning_method = pruning_method
        self.W_ref = W_ref  # 保存参考矩阵以供动态剪枝使用
        
        # 初始化权重矩阵 W
        if W_init is not None:
            if trainable:
                self.W = nn.Parameter(torch.tensor(W_init * 1e-5, dtype=torch.float32))
            else:
                self.register_buffer('W', torch.tensor(W_init * 1e-5, dtype=torch.float32))
        else:
            self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 1e-5)
        # 初始化时不自动剪枝

    def apply_hungarian_pruning(self):
        """动态应用匈牙利剪枝"""
        if self.pruning_method == "hungarian" and self.W_ref is not None:
            W_np = self.W.detach().cpu().numpy()
            W_ref_np = self.W_ref.cpu().numpy() if torch.is_tensor(self.W_ref) else self.W_ref
            cost_matrix = 1 - np.abs(W_np * W_ref_np)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            mask = torch.ones_like(self.W)
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] > 0.5:
                    mask[i, j] = 0
            prune.custom_from_mask(self, name='W', mask=mask)

    def forward(self, x):
        batch_size = x.size(0)
        r_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        E_t = self.input_to_hidden(x.view(batch_size, -1))
        r_t = torch.relu(r_t @ self.W + E_t + r_t)
        zero_input = torch.zeros(batch_size, self.input_size, device=x.device)
        for _ in range(9):
            E_t = self.input_to_hidden(zero_input)
            r_t = torch.relu(r_t @ self.W + E_t + r_t)
        return self.hidden_to_output(r_t)

class CWSRNN(BaseRNN):
    def __init__(self, input_size, hidden_size, output_size, C_init, 
                 train_W=True, train_C=False, non_zero_count=None):
        super().__init__(input_size, hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.non_zero_count = non_zero_count
        
        # 初始化 C 矩阵
        C_binary = torch.tensor(C_init != 0).float()
        if train_C:
            self.C = nn.Parameter(C_binary)
        else:
            self.register_buffer('C', C_binary)
        self.register_buffer('C_mask', C_binary.clone())
        
        # 初始化时不自动剪枝
        pos_ratio = 0.7
        num_pos = int(pos_ratio * hidden_size)
        s = torch.cat([torch.ones(num_pos), -torch.ones(hidden_size - num_pos)])
        self.register_buffer("s", s[torch.randperm(hidden_size)])

    def apply_drosophila_pruning(self):
        """动态应用果蝇剪枝"""
        if self.non_zero_count is None or not isinstance(self.C, nn.Parameter):
            return
        total_params = self.C.numel()
        prune_num = total_params - self.non_zero_count
        if prune_num > 0 and prune_num < total_params:
            flat_C = self.C.flatten() * self.C_mask.flatten()
            non_zero_indices = torch.nonzero(self.C_mask.flatten()).squeeze()
            if non_zero_indices.numel() > 0:
                values_at_non_zero = torch.abs(flat_C[non_zero_indices])
                if values_at_non_zero.numel() > prune_num:
                    _, indices_to_prune = torch.topk(values_at_non_zero, k=prune_num, largest=False)
                    prune_indices = non_zero_indices[indices_to_prune]
                    mask = torch.ones_like(flat_C)
                    mask[prune_indices] = 0
                    self.C.data = (self.C * mask.reshape(self.C.shape)).clamp(0, 1)

    def forward(self, x):
        W_eff = self.C.clamp(0, 1) * self.W * self.s.unsqueeze(1)
        batch_size = x.size(0)
        r_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        E_t = self.input_to_hidden(x.view(batch_size, -1))
        r_t = torch.relu(r_t @ W_eff + E_t + r_t)
        zero_input = torch.zeros(batch_size, self.input_size, device=x.device)
        for _ in range(9):
            E_t = self.input_to_hidden(zero_input)
            r_t = torch.relu(r_t @ W_eff + E_t + r_t)
        return self.hidden_to_output(r_t)

class CNNRNN(nn.Module):
    def __init__(self, W_init, conv_channels=16, time_steps=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.input_size = conv_channels * 14 * 14
        self.hidden_size = W_init.shape[0]
        self.output_size = 10
        self.input_to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.register_buffer('W', torch.tensor(W_init * 1e-5, dtype=torch.float32))
      
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        batch_size = x.size(0)
        r_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        E_t = self.input_to_hidden(conv_out)
        r_t = torch.relu(r_t @ self.W + E_t + r_t)
        zero_input = torch.zeros(batch_size, self.input_size, device=x.device)
        for _ in range(9):
            E_t = self.input_to_hidden(zero_input)
            r_t = torch.relu(r_t @ self.W + E_t + r_t)
        return self.hidden_to_output(r_t)

class SingleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden = self.input_to_hidden(x)
        output = self.hidden_to_output(hidden)
        return output

class TwohiddenMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_to_hidden1 = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden1_to_hidden2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hidden2_to_output = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden1 = self.relu(self.input_to_hidden1(x))
        hidden2 = self.relu(self.hidden1_to_hidden2(hidden1))
        output = self.hidden2_to_output(hidden2)
        return output

class StaticMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_to_hidden1 = nn.Linear(input_size, hidden_size, bias=True)
        # 注册固定权重矩阵（1360x1360）
        self.register_buffer('hidden1_to_hidden2', torch.randn(hidden_size, hidden_size))
        self.hidden2_to_output = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()
  
    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden1 = self.relu(self.input_to_hidden1(x))
        # 使用注册的权重矩阵
        hidden2 = self.relu(torch.matmul(hidden1, self.hidden1_to_hidden2))
        output = self.hidden2_to_output(hidden2)
        return output
    
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias=True)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output

class FISTAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, lambda_l1=1e-5):
        defaults = dict(lr=lr, lambda_l1=lambda_l1)
        super().__init__(params, defaults)
      
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.data = p.data - group['lr'] * p.grad
                p.data = torch.sign(p.data) * torch.clamp(torch.abs(p.data) - group['lr']*group['lambda_l1'], min=0)