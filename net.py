import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def load_drosophila_matrix(csv_path, apply_pruning=False):
    W_df = pd.read_csv(csv_path, index_col=0, header=0)
    W = W_df.values.astype(np.float32)
    W_min, W_max = W.min(), W.max()
    W_norm = (W - W_min) / (W_max - W_min + 1e-8)
  
    if apply_pruning:
        non_zero_count = np.count_nonzero(W)
        return W_norm, non_zero_count
    return W_norm

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
        self.W_ref = W_ref  
        
        if W_init is not None:
            if trainable:
                self.W = nn.Parameter(torch.tensor(W_init * 1e-5, dtype=torch.float32))
            else:
                self.register_buffer('W', torch.tensor(W_init * 1e-5, dtype=torch.float32))
        else:
            self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 1e-5)

    def apply_hungarian_pruning(self):
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
        
        C_binary = torch.tensor(C_init != 0).float()
        if train_C:
            self.C = nn.Parameter(C_binary)
        else:
            self.register_buffer('C', C_binary)
        self.register_buffer('C_mask', C_binary.clone())
        
        pos_ratio = 0.7
        num_pos = int(pos_ratio * hidden_size)
        s = torch.cat([torch.ones(num_pos), -torch.ones(hidden_size - num_pos)])
        self.register_buffer("s", s[torch.randperm(hidden_size)])

    def apply_drosophila_pruning(self):
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

# class TwohiddenMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.input_to_hidden1 = nn.Linear(input_size, hidden_size, bias=True)
#         self.hidden1_to_hidden2 = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.hidden2_to_output = nn.Linear(hidden_size, output_size, bias=True)

#         self.hidden1_to_hidden2.weight.requires_grad = False  # 冻结权重

#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         hidden1 = self.relu(self.input_to_hidden1(x))
#         hidden2 = self.relu(self.hidden1_to_hidden2(hidden1))
#         output = self.hidden2_to_output(hidden2)
#         return outputß

class TwoHiddenMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.input_to_hidden1 = nn.Linear(input_size, hidden_size)
        
        W_init = torch.randn(hidden_size, hidden_size) * 0.01
        self.register_buffer("hidden1_to_hidden2", W_init)
        
        self.hidden2_to_output = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 若 x 已经是 [batch_size, input_size] 可不 reshape
        hidden1 = self.relu(self.input_to_hidden1(x))
        
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