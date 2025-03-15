import torch
import torch.nn as nn
import math

class BasicRNN(nn.Module):
    def __init__(self, 
                 W_init,
                 input_dim: int,
                 sensory_dim: int,
                 internal_dim: int,
                 output_dim: int,
                 num_classes: int, 
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16):
        """
        Unifies W_ss, W_sr, W_rs, W_rr, W_ro, W_or, W_so, W_oo, W_os into one
        big matrix W of shape (S+I+O, S+I+O). We'll slice it for sub-blocks.
        
        LoRA parameters:
        - use_lora: Whether to use LoRA adaptation
        - lora_rank: Rank of the LoRA matrices (r in the paper)
        - lora_alpha: Scaling factor for LoRA (alpha in the paper)
        """
        super().__init__()
        print(f"BasicRNN init: trainable={trainable}, pruning={pruning}, target_nonzeros={target_nonzeros}, lambda_l1={lambda_l1}")
        print(f"LoRA config: use_lora={use_lora}, rank={lora_rank}, alpha={lora_alpha}")
        
        self.sensory_dim = sensory_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.total_dim = sensory_dim + internal_dim + output_dim

        self.pruning = pruning
        self.lambda_l1 = lambda_l1
        self.target_nonzeros = target_nonzeros
        
        print(f"W_init.shape: {W_init.shape}, sensory_dim: {sensory_dim}, internal_dim: {internal_dim}, output_dim: {output_dim}")
        assert W_init.shape[0] == self.total_dim
        
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
            # Initialize LoRA matrices A and B
            # A: [total_dim × rank], B: [rank × total_dim]
            self.lora_A = nn.Parameter(torch.zeros(self.total_dim, lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(lora_rank, self.total_dim))
            # Initialize with small random values
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)  # Initialize B to zero as in LoRA paper

        # Input projection (input -> S)
        self.input_proj = nn.Linear(input_dim, sensory_dim)
        # Output layer (O -> num_classes)
        self.output_layer = nn.Linear(output_dim, num_classes)
        self.activation = nn.ReLU()

    def get_effective_W(self):
        """Get the effective weight matrix including LoRA if enabled"""
        if not self.use_lora:
            return self.W
        
        # Compute LoRA contribution: (A × B) × scaling
        lora_contribution = (self.lora_A @ self.lora_B) * self.lora_scaling
        return self.W + lora_contribution

    def forward(self, x, time_steps=10):
        """
        Forward pass with states S, I, O. We slice the effective W into sub-blocks:
          W_ss, W_sr, W_so, W_rs, W_rr, W_ro, W_os, W_or, W_oo
        """
        batch_size, device = x.shape[0], x.device
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

        # Input projection
        E = self.input_proj(x)

        for t in range(time_steps):
            # Optionally only inject input every 5 steps
            E_t = E if (t % 5 == 0) else torch.zeros_like(E)

            S_next = self.activation(
                S_state @ W_ss + E_t + I_state @ W_rs + O_state @ W_os
            )
            I_next = self.activation(
                I_state @ W_rr + S_state @ W_sr + O_state @ W_or
            )
            O_next = self.activation(
                O_state @ W_oo + I_state @ W_ro + S_state @ W_so
            )

            S_state, I_state, O_state = S_next, I_next, O_next

        return self.output_layer(O_state)

    def get_l1_loss(self):
        """Compute L1 regularization loss on base weights only"""
        return self.W.abs().sum()

    def enforce_sparsity(self):
        """Hard threshold to maintain target sparsity level on base weights only"""
        with torch.no_grad():
            if self.target_nonzeros is None:
                return
                
            W_flat = self.W.view(-1)
            numel = W_flat.numel()
            
            # Sort by absolute value (descending)
            values, indices = torch.sort(W_flat.abs(), descending=True)
            
            # Get threshold value
            if self.target_nonzeros >= numel:
                return
            threshold = values[self.target_nonzeros]
            
            # Zero out values below threshold
            mask = (self.W.abs() >= threshold)
            self.W.data.mul_(mask.float())

class ThreeHiddenMLP(nn.Module):
    def __init__(self, input_size=784, hidden1_size=29, hidden2_size=147, hidden3_size=400, output_size=10, 
                 freeze=False):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.output_size = output_size

        self.input_to_hidden1 = nn.Linear(input_size, hidden1_size, bias=True)

        if freeze:
            # If frozen, register as buffer
            self.register_buffer('hidden1_to_hidden2', torch.randn(hidden1_size, hidden2_size))
            self.register_buffer('hidden2_to_hidden3', torch.randn(hidden2_size, hidden3_size))
        else:
            # If trainable, register as parameter
            self.hidden1_to_hidden2 = nn.Parameter(torch.randn(hidden1_size, hidden2_size))
            self.hidden2_to_hidden3 = nn.Parameter(torch.randn(hidden2_size, hidden3_size))

        self.hidden3_to_output = nn.Linear(hidden3_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input if needed
        x = x.view(x.size(0), -1)

        hidden1 = self.relu(self.input_to_hidden1(x))
        hidden2 = self.relu(torch.matmul(hidden1, self.hidden1_to_hidden2))
        hidden3 = self.relu(torch.matmul(hidden2, self.hidden2_to_hidden3))

        output = self.hidden3_to_output(hidden3)
        return output