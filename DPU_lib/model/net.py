import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self,
                 W_init,
                 sensory_dim: int,
                 internal_dim: int,
                 output_dim: int,
                 num_out: int,
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16,
                 drop_out=True,
                 dropout_rate: float = 0.2,
                 timesteps = 5,
                 filter_num = 1,
                 cumulate_output: bool = False,
                 use_residual: bool = False,
                 use_relu: bool = False,
                 ):
        super().__init__()
        self.use_relu = use_relu
        self.filter_num = filter_num
        self.convs = nn.ModuleList([
            nn.Conv2d(8, self.filter_num, kernel_size=k, stride=1, padding=0)
            for k in range(1, 9)
        ])
        self.cnn_output_dim = sum((8 - k + 1) ** 2 for k in range(1, 9)) * self.filter_num
        print(f"CNN output feature ct = {self.cnn_output_dim}")

        self.basicrnn = BasicRNN(
            W_init=W_init,
            input_dim=self.cnn_output_dim,
            sensory_dim=sensory_dim,
            internal_dim=internal_dim,
            output_dim=output_dim,
            num_out=num_out,
            trainable=trainable,
            pruning=pruning,
            target_nonzeros=target_nonzeros,
            lambda_l1=lambda_l1,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            drop_out = drop_out,
            dropout_rate=dropout_rate,
            timesteps=timesteps,
            cumulate_output=cumulate_output,
            use_residual=use_residual
        )


    def forward(self, x):
        if hasattr(self, 'use_relu') and self.use_relu:
            conv_outputs = [F.relu(conv(x.permute(0, 3, 1, 2))) for conv in self.convs]
        else:
            conv_outputs = [conv(x.permute(0, 3, 1, 2)) for conv in self.convs]

        conv_outputs = torch.cat([out.reshape(out.size(0), -1) for out in conv_outputs], dim=1)

        return self.basicrnn(conv_outputs)

class BasicRNN(nn.Module):
    def __init__(self, 
                 W_init,
                 input_dim: int,
                 sensory_dim: int,
                 internal_dim: int,
                 output_dim: int,
                 num_out: int,
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16,
                 drop_out = True,
                 dropout_rate: float = 0.2,
                 cumulate_output: bool = False,
                 use_residual: bool = False,
                 timesteps : int = 5,
                 ):
        """
        Unifies W_ss, W_sr, W_rs, W_rr, W_ro, W_or, W_so, W_oo, W_os into one
        big matrix W of shape (S+I+O, S+I+O). We'll slice it for sub-blocks.
        
        LoRA parameters:
        - use_lora: Whether to use LoRA adaptation
        - lora_rank: Rank of the LoRA matrices (r in the paper)
        - lora_alpha: Scaling factor for LoRA (alpha in the paper)
        
        Regularization parameters:
        - dropout_rate: Rate for dropout applied to the input layer
        """
        super().__init__()
        
        print(f"BasicRNN init: trainable={trainable}, pruning={pruning}, target_nonzeros={target_nonzeros}, lambda_l1={lambda_l1}")
        print(f"LoRA config: use_lora={use_lora}, rank={lora_rank}, alpha={lora_alpha}")
        print(f"Regularization: dropout_rate={dropout_rate}" if drop_out else "No Dropout")
        
        self.sensory_dim = sensory_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.total_dim = sensory_dim + internal_dim + output_dim
        
        self.pruning = pruning
        self.lambda_l1 = lambda_l1
        self.target_nonzeros = target_nonzeros

        self.cumulate_output = cumulate_output
        self.use_residual = use_residual
        if self.cumulate_output:
            assert self.use_residual == False
        
        print(f"W_init.shape: {W_init.shape}, sensory_dim: {sensory_dim}, internal_dim: {internal_dim}, output_dim: {output_dim}")
        assert W_init.shape[0] == self.total_dim
        
        # Store initial weights and sparsity mask for similarity comparison
        self.register_buffer('W_init', torch.tensor(W_init, dtype=torch.float32))
        self.register_buffer('sparsity_mask', torch.tensor(W_init != 0, dtype=torch.float32))
        
        # Initialize base weight matrix (frozen DPU weights)
        W_init_tensor = torch.tensor(W_init, dtype=torch.float32)
        if trainable:
            self.W = nn.Parameter(W_init_tensor)
        else:
            self.register_buffer('W', W_init_tensor)

        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank

        if use_lora:
            # Initialize LoRA matrices A and B with improved initialization
            self.lora_A = nn.Parameter(torch.empty(self.total_dim, lora_rank))
            self.lora_B = nn.Parameter(torch.empty(lora_rank, self.total_dim))
            
            # Use SVD-based initialization for better starting point
            U, S, V = torch.svd(W_init_tensor)
            # Initialize A with first r singular vectors
            self.lora_A.data.copy_(U[:, :lora_rank] * torch.sqrt(S[:lora_rank].unsqueeze(0)))
            # Initialize B with first r singular vectors
            self.lora_B.data.copy_(torch.sqrt(S[:lora_rank].unsqueeze(1)) * V[:, :lora_rank].t())
            # Scale B to make initial LoRA contribution small
            self.lora_B.data.mul_(2.0)

        self.input_proj = nn.Linear(input_dim, sensory_dim)
        self.output_layer = nn.Linear(output_dim, num_out)
        self.activation = nn.ReLU()
        
        # Dropout layer for regularization
        self.use_dropout = drop_out
        self.dropout = nn.Dropout(dropout_rate)

        # set timesteps
        self.timesteps = timesteps

    def forward(self, x):
        """
        Forward pass with states S, I, O. We slice the effective W into sub-blocks:
          W_ss, W_sr, W_so, W_rs, W_rr, W_ro, W_os, W_or, W_oo
        """
        batch_size, device = x.shape[0], x.device
        
        # Just flatten the input
        x = x.view(batch_size, -1)

        # Get effective weight matrix (base + LoRA if enabled)
        W_eff = self.get_effective_W()

        # Partition the effective matrix W
        S, I, O = self.sensory_dim, self.internal_dim, self.output_dim
        W_ss = W_eff[0:S,   0:S]
        W_sr = W_eff[0:S,   S:S+I]
        W_so = W_eff[0:S,   S+I:S+I+O]
        W_rs = W_eff[S:S+I, 0:S]
        W_rr = W_eff[S:S+I, S:S+I]
        W_ro = W_eff[S:S+I, S+I:S+I+O]
        W_os = W_eff[S+I:S+I+O, 0:S]
        W_or = W_eff[S+I:S+I+O, S:S+I]
        W_oo = W_eff[S+I:S+I+O, S+I:S+I+O]

        # Initialize states S, I, O to zero
        S_state = torch.zeros(batch_size, S, device=device)
        I_state = torch.zeros(batch_size, I, device=device)
        O_state = torch.zeros(batch_size, O, device=device)

        E = self.input_proj(x)

        # patches for some previous saved models do not have this attribute
        # this part can be cleaned up a bit but not necessary....
        if hasattr(self, 'use_dropout'):
            if self.use_dropout:
                E = self.dropout(E)
        else:
            E = self.dropout(E)
        if not hasattr(self, 'cumulate_output'):
            self.cumulate_output = False
        if not hasattr(self, 'use_residual'):
            self.use_residual = False

        if self.cumulate_output:
            cumulate_output = torch.zeros(batch_size, O, device=device)

        for t in range(self.timesteps):
            # single injection only
            E_t = E if (t == 0) else torch.zeros_like(E)

            S_next = S_state @ W_ss + E_t + I_state @ W_rs + O_state @ W_os
            I_next = I_state @ W_rr + S_state @ W_sr + O_state @ W_or
            O_next = O_state @ W_oo + I_state @ W_ro + S_state @ W_so
            if self.use_residual:
                S_next = S_state + S_next
                I_next = I_state + I_next
                O_next = O_state + O_next

            S_next = self.activation(S_next)
            I_next = self.activation(I_next)
            O_next = self.activation(O_next)

            S_state, I_state, O_state = S_next, I_next, O_next

            if self.cumulate_output:
                cumulate_output = O_state + cumulate_output

        if self.cumulate_output:
            out = self.output_layer(cumulate_output)
        else:
            out = self.output_layer(O_state)
        return out

    def get_effective_W(self):
        """Get the effective weight matrix including LoRA if enabled"""
        if not self.use_lora:
            return self.W

        # Compute LoRA contribution: (A × B) × scaling
        lora_contribution = (self.lora_A @ self.lora_B) * self.lora_scaling

        # Combine base weights with LoRA contribution
        effective_W = self.W + lora_contribution

        # Apply sparsity mask to maintain sparsity pattern
        return effective_W * self.sparsity_mask







######>>>>>>>>>>>>>>>>>>>>CURRENT VERSION
class BasicRNN_new(nn.Module):
    def __init__(self,
                 W_init,
                 input_dim: int,
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,
                 num_out: int,
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16,
                 drop_out=True,
                 dropout_rate: float = 0.2,
                 cumulate_output: bool = False,
                 use_residual: bool = False,
                 timesteps: int = 5,
                 learnable_KC = None
                 ):
        super().__init__()

        print(
            f"BasicRNN init: trainable={trainable}, pruning={pruning}, target_nonzeros={target_nonzeros}, lambda_l1={lambda_l1}")
        print(f"LoRA config: use_lora={use_lora}, rank={lora_rank}, alpha={lora_alpha}")
        print(f"Regularization: dropout_rate={dropout_rate}" if drop_out else "No Dropout")

        self.sensory_idx = sensory_idx
        self.KC_idx = KC_idx
        self.internal_idx = internal_idx
        self.output_idx = output_idx
        self.total_dim = len(self.sensory_idx) + len(self.KC_idx) + len(self.internal_idx) + len(self.output_idx)

        self.pruning = pruning
        self.lambda_l1 = lambda_l1
        self.target_nonzeros = target_nonzeros

        self.cumulate_output = cumulate_output
        self.use_residual = use_residual
        if self.cumulate_output:
            assert self.use_residual == False

        W_init_tensor = torch.tensor(W_init, dtype=torch.float32)
        if trainable:
            mask = torch.zeros_like(W_init_tensor)
            if learnable_KC == 'between':# train any connection that has one end in KC, changing weights = 3244
                non_zero_mask = torch.tensor(W_init != 0, dtype=torch.float32)
                mask[self.KC_idx, :] = non_zero_mask[self.KC_idx, :]
                mask[:, self.KC_idx] = non_zero_mask[:, self.KC_idx]
            elif learnable_KC == 'within': # train the whole 144 * 144, changing weights = 20736
                KC_idx_tensor = torch.tensor(self.KC_idx)
                rows, cols = torch.meshgrid(KC_idx_tensor, KC_idx_tensor, indexing='ij')
                mask[rows, cols] = 1.0
            else: # train all non-zero in the W
                mask = torch.tensor(W_init != 0, dtype=torch.float32)
            self.register_buffer('mask', mask)

            self.W = nn.Parameter(W_init_tensor)
            self.W.register_hook(lambda grad: grad * self.mask)
        else:
            self.register_buffer('W', W_init_tensor)

        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank

        if use_lora:
            # Initialize LoRA matrices A and B with improved initialization
            self.lora_A = nn.Parameter(torch.empty(self.total_dim, lora_rank))
            self.lora_B = nn.Parameter(torch.empty(lora_rank, self.total_dim))

            # Use SVD-based initialization for better starting point
            U, S, V = torch.svd(W_init_tensor)
            # Initialize A with first r singular vectors
            self.lora_A.data.copy_(U[:, :lora_rank] * torch.sqrt(S[:lora_rank].unsqueeze(0)))
            # Initialize B with first r singular vectors
            self.lora_B.data.copy_(torch.sqrt(S[:lora_rank].unsqueeze(1)) * V[:, :lora_rank].t())
            # Scale B to make initial LoRA contribution small
            self.lora_B.data.mul_(2.0)

        self.input_proj = nn.Linear(input_dim, len(self.sensory_idx))
        self.output_layer = nn.Linear(len(self.output_idx), num_out)
        self.activation = nn.ReLU()

        # Dropout layer for regularization
        self.use_dropout = drop_out
        self.dropout = nn.Dropout(dropout_rate)

        # set timesteps
        self.timesteps = timesteps

    def forward(self, x):
        batch_size, device = x.shape[0], x.device
        x = x.view(batch_size, -1)
        E = self.input_proj(x)

        if hasattr(self, 'use_dropout'):
            if self.use_dropout:
                E = self.dropout(E)
        else:
            E = self.dropout(E)
        E = F.pad(E, (0, self.total_dim - E.size(1)), mode='constant', value=0)

        h_state = torch.zeros(batch_size, self.total_dim, device=device)

        W_eff = self.get_effective_W()

        for t in range(self.timesteps):
            E_t = E if (t == 0) else torch.zeros_like(E)
            h_next = h_state @ W_eff + E_t
            if self.use_residual:
                h_next = h_state + h_next
            h_next = self.activation(h_next)
            h_state = h_next
        out = self.output_layer(h_state[:, self.output_idx])
        return out

    def get_effective_W(self):
        """Get the effective weight matrix including LoRA if enabled"""
        if not self.use_lora:
            return self.W

        # Compute LoRA contribution: (A × B) × scaling
        lora_contribution = (self.lora_A @ self.lora_B) * self.lora_scaling

        # Combine base weights with LoRA contribution
        effective_W = self.W + lora_contribution

        # Apply sparsity mask to maintain sparsity pattern
        return effective_W * self.sparsity_mask



class BasicCNN_new(nn.Module):
    def __init__(self,
                 W_init,
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,
                 num_out: int,
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16,
                 drop_out=True,
                 dropout_rate: float = 0.2,
                 timesteps = 5,
                 filter_num = 1,
                 cumulate_output: bool = False,
                 use_residual: bool = False,
                 use_relu: bool = False,
                 learnable_KC = None
                 ):
        super().__init__()
        self.use_relu = use_relu
        self.filter_num = filter_num
        self.convs = nn.ModuleList([
            nn.Conv2d(8, self.filter_num, kernel_size=k, stride=1, padding=0)
            for k in range(1, 9)
        ])
        self.cnn_output_dim = sum((8 - k + 1) ** 2 for k in range(1, 9)) * self.filter_num
        print(f"CNN output feature ct = {self.cnn_output_dim}")

        self.basicrnn = BasicRNN_new(
            W_init=W_init,
            input_dim=self.cnn_output_dim,
            sensory_idx=sensory_idx,
            KC_idx=KC_idx,
            internal_idx=internal_idx,
            output_idx=output_idx,
            num_out=num_out,
            trainable=trainable,
            pruning=pruning,
            target_nonzeros=target_nonzeros,
            lambda_l1=lambda_l1,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            drop_out = drop_out,
            dropout_rate=dropout_rate,
            timesteps=timesteps,
            cumulate_output=cumulate_output,
            use_residual=use_residual,
            learnable_KC = learnable_KC,
        )


    def forward(self, x):
        if hasattr(self, 'use_relu') and self.use_relu:
            conv_outputs = [F.relu(conv(x.permute(0, 3, 1, 2))) for conv in self.convs]
        else:
            conv_outputs = [conv(x.permute(0, 3, 1, 2)) for conv in self.convs]

        conv_outputs = torch.cat([out.reshape(out.size(0), -1) for out in conv_outputs], dim=1)

        return self.basicrnn(conv_outputs)

