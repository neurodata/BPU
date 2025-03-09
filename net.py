import torch
import math
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def init_param(obj, name, value, trainable):
    """Initialize a weight as a parameter (if trainable) or as a buffer."""
    tensor = torch.tensor(value, dtype=torch.float32)
    if trainable:
        setattr(obj, name, nn.Parameter(tensor, requires_grad=True))
    else:
        obj.register_buffer(name, tensor)

class DrosophilaRNN(nn.Module):
    def __init__(self, input_dim, num_classes,
                 conn_weights, ref_weights=None, trainable=True, pruning=True, SIO=False, 
                 pruning_cfg=None
        ):
        super().__init__()
        self.SIO = SIO
        self.pruning = pruning
        self.ref_weights = ref_weights

        self.pruning_constraint = pruning_cfg.get('constraint', 'structure')
        self.lambda_reg = pruning_cfg.get('lambda_reg', 0.0002)
        self.max_iter = pruning_cfg.get('max_iter', 200)
        self.fista_threshold = pruning_cfg.get('fista_threshold', 0.3)
        self.fista_gamma = pruning_cfg.get('fista_gamma', 0.6)
        self.target_nonzeros = pruning_cfg.get('target_nonzeros', {'W_ss': 0, 'W_sr': 213, 'W_rs': 0, 'W_rr': 61782})
        self.rescale_factor=pruning_cfg.get('rescale_factor', 0)
        if SIO:
            self.sensory_dim = conn_weights['W_ss'].shape[0]
            self.internal_dim = conn_weights['W_rr'].shape[0]
            self.output_dim = conn_weights['W_oo'].shape[0]
            # Initialize SIO weight matrices
            for name in ["W_ss", "W_sr", "W_rs", "W_rr", "W_ro", "W_or", "W_so", "W_oo", "W_os"]:
                init_param(self, name, conn_weights[name], trainable)
            if pruning and ref_weights is not None:
                for name in ["W_ss", "W_sr", "W_rs", "W_rr", "W_ro", "W_or", "W_so", "W_oo", "W_os"]:
                    setattr(self, f"{name}_ref", ref_weights[name])
            self.input_proj = nn.Linear(input_dim, self.sensory_dim)
            self.output_layer = nn.Linear(self.output_dim, num_classes)
            self.activation = nn.ReLU()
        else:
            self.sensory_dim = conn_weights['W_ss'].shape[0]
            self.residual_dim = conn_weights['W_rr'].shape[0]
            for name in ["W_ss", "W_sr", "W_rs", "W_rr"]:
                init_param(self, name, conn_weights[name], trainable)
            if pruning and ref_weights is not None:
                for name in ["W_ss", "W_sr", "W_rs", "W_rr"]:
                    setattr(self, f"{name}_ref", ref_weights[name])
            self.input_proj = nn.Linear(input_dim, self.sensory_dim)
            self.output_layer = nn.Linear(self.residual_dim, num_classes)
            self.activation = nn.ReLU()

    def forward(self, x, time_steps=10):
        if self.SIO:
            batch_size, device = x.shape[0], x.device
            x = x.view(x.size(0), -1)
            S = torch.zeros(batch_size, self.sensory_dim, device=device)
            R = torch.zeros(batch_size, self.internal_dim, device=device)
            O = torch.zeros(batch_size, self.output_dim, device=device)
            E = self.input_proj(x)
            for t in range(time_steps):
                E_t = E if t % 5 == 0 else torch.zeros_like(E)
                S_next = self.activation(S @ self.W_ss + E_t + R @ self.W_rs + O @ self.W_os)
                R_next = self.activation(R @ self.W_rr + S @ self.W_sr + O @ self.W_or)
                O_next = self.activation(O @ self.W_oo + R @ self.W_ro + S @ self.W_so)
                S, R, O = S_next, R_next, O_next
            return self.output_layer(O)
        else:
            batch_size, device = x.shape[0], x.device
            x = x.view(x.size(0), -1)
            S = torch.zeros(batch_size, self.W_ss.shape[0], device=device)
            R = torch.zeros(batch_size, self.W_rr.shape[0], device=device)
            E = self.input_proj(x)
            for t in range(time_steps):
                E_t = E if t % 5 == 0 else torch.zeros_like(E)
                S_next = self.activation(S @ self.W_ss + E_t + R @ self.W_rs)
                R_next = self.activation(R @ self.W_rr + S @ self.W_sr)
                S, R = S_next, R_next
            return self.output_layer(R)
        
    def evaluate_aggregate_similarity(self, X_list, W_ref_list):
        """
        Compute an aggregated similarity across multiple pruned submodules.
        We'll sum the overlap, union, L1, L2, etc. across them all.
        """
        total_overlap = 0
        total_union = 0
        total_nz_X = 0
        total_nz_Wref = 0
        total_L1 = 0.0
        total_L2_sqr = 0.0

        for X, W_ref in zip(X_list, W_ref_list):
            X_nz = np.count_nonzero(X)
            Wref_nz = np.count_nonzero(W_ref)
            overlap_nz = np.count_nonzero((X != 0) & (W_ref != 0))
            union_nz = np.count_nonzero((X != 0) | (W_ref != 0))

            total_nz_X += X_nz
            total_nz_Wref += Wref_nz
            total_overlap += overlap_nz
            total_union += union_nz

            diff = X - W_ref
            total_L1 += np.abs(diff).sum()
            total_L2_sqr += (diff**2).sum()

        overlap_ratio = total_overlap / total_union if total_union > 0 else 0.0
        L2 = math.sqrt(total_L2_sqr)

        similarity_dict = {
            'overlap_ratio': overlap_ratio,
            'nonzero_count_X': total_nz_X,
            'nonzero_count_Wref': total_nz_Wref,
            'overlap_positions': total_overlap,
            'L1_error': total_L1,
            'L2_error': L2
        }
        print(f"\n Similarity Metrics: Overlap ratio: {overlap_ratio:.4f}, Non-zero count of X: {total_nz_X}, Non-zero count of W_ref: {total_nz_Wref}, Overlap of non-zero positions: {total_overlap}, L1 error: {total_L1:.4f}, L2 error: {L2:.4f}")
        return similarity_dict
    
    def apply_structure_constraint_pruning(self):
        names = (["W_ss", "W_sr", "W_rs", "W_rr", "W_ro", "W_or", "W_so", "W_oo"]
                 if self.SIO else ["W_ss", "W_sr", "W_rs", "W_rr"])
        pruned_list = []
        wref_list = []

        for name in names:
            W_tensor = getattr(self, name, None)
            W_ref = getattr(self, f"{name}_ref", None)

            if W_tensor is None or W_ref is None:
                continue

            W_np = W_tensor.detach().cpu().numpy()
            w = np.where(W_ref != 0, 0.1, 2.0)

            lambda_reg = self.lambda_reg
            max_iter = self.max_iter
            t = 0.5

            # c_{ij} = -1 if W_ref[i,j] != 0 else +1
            c = np.where(W_ref != 0, -1.0, 1.0)

            M0 = np.ones_like(W_np)
            M = M0.copy()
            Y = M0.copy()
            t_k = 1.0

            def grad_f(M, W, W_ref, gamma, c):
                """
                f(M) = 0.5 * || (M⊙W) - W_ref ||_F^2   +   gamma * sum_{i,j} c_{i,j} * M_{i,j}
                => grad_f(M) = ((M*W) - W_ref) * W   +   gamma*c
                """
                grad_data = (M * W - W_ref) * W
                return grad_data + gamma * c

            def prox_weighted_l1(Z, tau, lam, w):
                """
                g(M) = lam * sum_{i,j} [ w_{ij} * |M_{i,j}| ]
                => prox(Z) = sign(Z) * max(|Z| - tau*lam*w, 0)
                """
                return np.sign(Z) * np.maximum(np.abs(Z) - tau * lam * w, 0)

            def objective(M, W, W_ref, lambda_reg, w, gamma, c):
                """
                F(M) = 0.5 * ||(M⊙W) - W_ref||_F^2
                    + gamma * sum_{i,j} c_{i,j} * M_{i,j}
                    + lambda_reg * sum_{i,j} [ w_{i,j} * |M_{i,j}| ]
                """
                f_val = 0.5 * np.linalg.norm(M * W - W_ref, 'fro') ** 2
                mismatch_val = gamma * np.sum(c * M)
                g_val = lambda_reg * np.sum(w * np.abs(M))
                return f_val + mismatch_val + g_val

            def remove_negative_zeros(mat, eps=1e-9):
                mat_ = np.copy(mat)
                mat_[np.abs(mat_) < eps] = 0.0
                return mat_

            for k in range(max_iter):
                G = grad_f(Y, W_np, W_ref, self.fista_gamma, c)
                Z = Y - t * G
                M_next = prox_weighted_l1(Z, t, lambda_reg, w)
                # projection to [0,1]
                M_next = np.clip(M_next, 0, 1)

                # update FISTA momentum
                t_k_next = (1 + np.sqrt(1 + 4 * (t_k ** 2))) / 2.0
                Y = M_next + ((t_k - 1) / t_k_next) * (M_next - M)

                M = M_next
                t_k = t_k_next

            threshold = self.fista_threshold
            M_binary = (M > threshold).astype(np.float32)
            
            X = M_binary * W_np

            X_rounded = np.round(X, 2)
            X_fixed = remove_negative_zeros(X_rounded)

            mask_torch = torch.from_numpy(M_binary).to(W_tensor.device)
            prune.custom_from_mask(self, name=name, mask=mask_torch)
            prune.remove(self, name)
            with torch.no_grad():
                W_tensor.copy_(torch.from_numpy(X_fixed).to(W_tensor.device))

            pruned_list.append(X_fixed)
            wref_list.append(W_ref)
        return self.evaluate_aggregate_similarity(pruned_list, wref_list)


    def apply_sparsity_constraint_pruning(self):
        names = (["W_ss", "W_sr", "W_rs", "W_rr", "W_ro", "W_or", "W_so", "W_oo"]
                 if self.SIO else ["W_ss", "W_sr", "W_rs", "W_rr"])
        final_nonzeros = 0
        total = 0

        for name in names:
            target_nonzero = self.target_nonzeros.get(name, 0)
            W_tensor = getattr(self, name, None)
            W_ref = getattr(self, f"{name}_ref", None)

            if W_tensor is None or W_ref is None:
                continue

            W_np = W_tensor.detach().cpu().numpy()

            w = np.ones_like(W_np)
            
            lambda_reg = self.lambda_reg
            max_iter = self.max_iter
            t = 0.5

            M0 = np.ones_like(W_np)
            M = M0.copy()
            Y = M0.copy()
            t_k = 1.0  # momentum parameter
            

            # 4) Define gradient of the data fidelity:
            #    f(M) = 0.5 * || (M * W) - W ||_F^2
            # => grad_f(M) = ((M * W) - W) * W
            def grad_f(M, W):
                return (M * W - W) * W

            # 5) Define the proximal operator for weighted L1:
            #    g(M) = lambda_reg * sum w[i,j] * |M[i,j]|
            # => prox(Z) = sign(Z) * max(|Z| - step_size * lambda_reg * w, 0)
            def prox_weighted_l1(Z, step, lam, w_):
                return np.sign(Z) * np.clip(np.abs(Z) - step * lam * w_, a_min=0.0, a_max=None)

            def remove_negative_zeros(mat, eps=1e-9):
                mat_ = np.copy(mat)
                mat_[np.abs(mat_) < eps] = 0.0
                return mat_
            
            # ========== FISTA main loop ==========
            for k in range(max_iter):
                # Gradient of data fidelity at Y
                G = grad_f(Y, W_np)
                # Gradient descent
                Z = Y - t * G
                # Proximal step
                M_next = prox_weighted_l1(Z, t, lambda_reg, w)
                # Clamp M in [0,1]
                M_next = np.clip(M_next, 0, 1)

                # FISTA momentum update (using math.sqrt to avoid Torch error)
                t_k_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                Y = M_next + ((t_k - 1.0) / t_k_next) * (M_next - M)

                M = M_next
                t_k = t_k_next

            M_flat = M.reshape(-1)
            total_elems = M_flat.shape[0]

            if target_nonzero >= total_elems:
                thresh = -float("inf")
            elif target_nonzero == 0:
                print(f"Skipping pruning because target_nonzero is 0")
                continue  
            else:
                kth_index = max(0, min(total_elems - target_nonzero, total_elems - 1))
                thresh_val = np.partition(M_flat, kth_index)[kth_index]
                thresh = float(thresh_val)
            
            # Build the binary mask
            M_binary = (M >= thresh).astype(np.float32)   

            # If there's an off-by-one in top-k selection, we fix it
            # (e.g. if multiple elements == thresh)
            current_nz = int(np.count_nonzero(M_binary))

            if current_nz > target_nonzero:
                idx_ones = np.argwhere(M_binary == 1) 
                values = M[idx_ones[:, 0], idx_ones[:, 1]] 
                sorted_idx = np.argsort(values) 
                extra = current_nz - target_nonzero
                for i in range(extra):
                    index_in_ones = sorted_idx[i]  
                    row, col = idx_ones[index_in_ones] 
                    M_binary[row, col] = 0.0 

            X = M_binary * W_np

            X_rounded = np.round(X, 2)
            X_fixed = remove_negative_zeros(X_rounded)

            mask_torch = torch.from_numpy(M_binary).to(W_tensor.device)
            prune.custom_from_mask(self, name=name, mask=mask_torch)
            prune.remove(self, name)

            # Print final stats
            final_nz = int(np.count_nonzero(M_binary))  
            total_size = M_binary.size  

            with torch.no_grad():
                W_tensor.copy_(torch.from_numpy(X_fixed).to(W_tensor.device))
            
            final_nonzeros = final_nonzeros + int(np.count_nonzero(M_binary))
            total = total + M_binary.size

        similarity_dict = {
            'final_nonzeros': final_nonzeros,
            'total': total
        }
        print(f"\nFinal mask has {final_nz} nonzeros out of {total_size} "
                f"({final_nz / total_size * 100:.4f}%).")
        return similarity_dict

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
                if cost_matrix[i, j] > 0.99:
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

        self.selected_indices = torch.randint(0, input_size, (29,)) 
        self.mask = torch.zeros(input_size)  # mask
        self.mask[self.selected_indices] = 1  # activate 29 input
        
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        
        x_masked = x * self.mask.to(x.device) 
        
        hidden = self.input_to_hidden(x_masked)
        output = self.hidden_to_output(hidden)
        
        return output

class TwohiddenMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Select fixed 29 input features
        self.selected_indices = torch.randint(0, input_size, (29,))
        self.mask = torch.zeros(input_size)
        self.mask[self.selected_indices] = 1  # Only activate the selected 29 inputs

        self.input_to_hidden1 = nn.Linear(input_size, 29, bias=True)
        self.hidden1_to_hidden2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hidden2_to_output = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Apply the fixed mask to select the same 29 input features
        x_masked = x * self.mask.to(x.device)

        # First hidden layer
        hidden1 = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        hidden1[:, :29] = self.relu(self.input_to_hidden1(x_masked))

        # Second hidden layer
        hidden2 = self.relu(self.hidden1_to_hidden2(hidden1))
        output = self.hidden2_to_output(hidden2)

        return output


class StaticMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Select fixed 29 input features
        self.selected_indices = torch.randint(0, input_size, (29,))
        self.mask = torch.zeros(input_size)
        self.mask[self.selected_indices] = 1  # Only activate the selected 29 inputs

        self.input_to_hidden1 = nn.Linear(input_size, 29, bias=True)
        
        # Register a fixed weight matrix (hidden_size × hidden_size)
        self.register_buffer('hidden1_to_hidden2', torch.rand(hidden_size, hidden_size))

        self.hidden2_to_output = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Apply the fixed mask to select the same 29 input features
        x_masked = x * self.mask.to(x.device)

        # First hidden layer
        hidden1 = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        hidden1[:, :29] = self.relu(self.input_to_hidden1(x_masked))

        # Use registered weight matrix for transformation
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