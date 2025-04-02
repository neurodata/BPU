import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

class BasicRNN(nn.Module):
    def __init__(self, 
                 W_init,
                 input_dim: int,
                 sensory_dim: int,
                 internal_dim: int,
                 output_dim: int,
                 num_classes: int, 
                 sio: bool = True,
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16,
                 dropout_rate: float = 0.2,
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
        print(f"Regularization: dropout_rate={dropout_rate}")
        
        self.sensory_dim = sensory_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.total_dim = sensory_dim + internal_dim + output_dim
        self.sio = sio
        
        self.pruning = pruning
        self.lambda_l1 = lambda_l1
        self.target_nonzeros = target_nonzeros
        
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
        self.output_layer = nn.Linear(output_dim, num_classes)
        self.activation = nn.ReLU()
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
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

    def calculate_matrix_similarity(self):
        """
        Calculate similarity metrics between initial and current effective weight matrix.
        """
        W_eff = self.get_effective_W()
        W_init = self.W_init
        
        # Cosine similarity
        W_eff_flat = W_eff.flatten()
        W_init_flat = W_init.flatten()
        cosine_sim = F.cosine_similarity(W_eff_flat.unsqueeze(0), W_init_flat.unsqueeze(0))
        
        # Frobenius norm of difference
        frob_diff = torch.norm(W_eff - W_init, p='fro')
        
        # Relative Frobenius norm
        rel_frob_diff = frob_diff / torch.norm(W_init, p='fro')
        
        # Sparsity comparison
        init_sparsity = (W_init == 0).float().mean()
        eff_sparsity = (W_eff == 0).float().mean()
        
        # Additional sparsity metrics
        shared_nonzeros = ((W_init != 0) & (W_eff != 0)).float().mean()
        new_nonzeros = ((W_init == 0) & (W_eff != 0)).float().mean()
        
        return {
            'cosine_similarity': cosine_sim.item(),
            'frobenius_diff': frob_diff.item(),
            'relative_frobenius_diff': rel_frob_diff.item(),
            'init_sparsity': init_sparsity.item(),
            'effective_sparsity': eff_sparsity.item(),
            'shared_nonzeros': shared_nonzeros.item(),
            'new_nonzeros': new_nonzeros.item()
        }

    def forward(self, x, time_steps=2):
        """
        Forward pass with states S, I, O. We slice the effective W into sub-blocks:
          W_ss, W_sr, W_so, W_rs, W_rr, W_ro, W_os, W_or, W_oo
        """
        batch_size, device = x.shape[0], x.device
        
        # Just flatten the input
        x = x.view(batch_size, -1)

        # Get effective weight matrix (base + LoRA if enabled)
        W_eff = self.get_effective_W()

        if self.sio:
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

            # Input projection with dropout
            E = self.dropout(self.input_proj(x))

            for t in range(time_steps):
                # Optionally only inject input every 2 steps
                E_t = E if (t % 2 == 0) else torch.zeros_like(E)

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
        else:
            # Initialize state to zero
            state = torch.zeros(batch_size, self.total_dim, device=device)
            
            # Input projection with dropout
            E = self.dropout(self.input_proj(x))
            
            for t in range(time_steps):
                # Optionally only inject input every 2 steps
                E_t = E if (t % 2 == 0) else torch.zeros_like(E)
                
                # Update state using the whole matrix W
                # E_t is projected to the full dimension
                state_next = self.activation(
                    state @ W_eff + E_t
                )
                state = state_next
            
            # Use the whole state for output
            return self.output_layer(state)

    def get_l1_loss(self):
        """Compute L1 regularization loss on base weights only"""
        return self.W.abs().sum()

    def enforce_sparsity(self):
        """Hard threshold to maintain target sparsity level on base weights only"""
        with torch.no_grad():
            if self.target_nonzeros is None:
                return
                
            # Use reshape(-1) instead of view(-1) to handle non-contiguous tensors
            # This makes it work with column-permuted and other non-contiguous initializations
            W_flat = self.W.reshape(-1)
            numel = W_flat.numel()
            
            values, indices = torch.sort(W_flat.abs(), descending=True)
            if self.target_nonzeros >= numel:
                return
            threshold = values[self.target_nonzeros]
            
            # Zero out values below threshold
            mask = (self.W.abs() >= threshold)
            self.W.data.mul_(mask.float())

    def save_model(self, path, filename, metadata=None):
        """
        Save model and its configuration to a file.
        
        Parameters:
        -----------
        path : str
            Directory to save the model
        filename : str
            Base filename to use for saving
        metadata : dict, optional
            Additional metadata to save
        """
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, f'{filename}.pt'))
        
        # Save model configuration
        config = {
            'input_dim': self.input_proj.in_features,
            'sensory_dim': self.sensory_dim,
            'internal_dim': self.internal_dim,
            'output_dim': self.output_dim,
            'num_classes': self.output_layer.out_features,
            'W_init': self.W_init.cpu().numpy(),
            'trainable': isinstance(self.W, nn.Parameter),
            'pruning': self.pruning,
            'target_nonzeros': self.target_nonzeros,
            'lambda_l1': self.lambda_l1,
            'use_lora': self.use_lora,
            'lora_rank': getattr(self, 'lora_rank', 8),
            'lora_alpha': getattr(self, 'lora_alpha', 16),
            'sensory_type': getattr(self, 'sensory_type', 'visual'),
            'dropout_rate': getattr(self, 'dropout', nn.Dropout(0.2)).p,
            'use_position_encoding': getattr(self, 'use_position_encoding', False)
        }
        
        with open(os.path.join(path, 'model_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
            
        # Save additional metadata if provided
        if metadata:
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)

    @classmethod
    def load_model(cls, path, device=None):
        """
        Load a saved model from a directory.
        """
        # Load model configuration
        with open(os.path.join(path, 'model_config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        # Update device if provided
        if device:
            config['device'] = device
            
        # Extract parameters
        input_dim = config.get('input_dim')
        sensory_dim = config.get('sensory_dim')
        internal_dim = config.get('internal_dim')
        output_dim = config.get('output_dim')
        num_classes = config.get('num_classes')
        W_init = config.get('W_init')
        trainable = config.get('trainable', False)
        pruning = config.get('pruning', False)
        target_nonzeros = config.get('target_nonzeros', None)
        lambda_l1 = config.get('lambda_l1', 1e-4)
        device = config.get('device', 'cpu')
        use_lora = config.get('use_lora', False)
        lora_rank = config.get('lora_rank', 8)
        lora_alpha = config.get('lora_alpha', 16)
        sensory_type = config.get('sensory_type', 'visual')
        dropout_rate = config.get('dropout_rate', 0.2)
        use_position_encoding = config.get('use_position_encoding', False)
        
        # Create a new instance with the loaded parameters
        model = cls(
            W_init=W_init,
            input_dim=input_dim,
            sensory_dim=sensory_dim,
            internal_dim=internal_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            trainable=trainable,
            pruning=pruning,
            target_nonzeros=target_nonzeros,
            lambda_l1=lambda_l1,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sensory_type=sensory_type,
            dropout_rate=dropout_rate,
            use_position_encoding=use_position_encoding
        )
        
        # Load the model state
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=device))
        model.to(device)
        
        # Load additional metadata if available
        metadata_path = os.path.join(path, 'metadata.pkl')
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
        return model, metadata

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
        x = x.view(x.size(0), -1)

        hidden1 = self.relu(self.input_to_hidden1(x))
        hidden2 = self.relu(torch.matmul(hidden1, self.hidden1_to_hidden2))
        hidden3 = self.relu(torch.matmul(hidden2, self.hidden2_to_hidden3))

        output = self.hidden3_to_output(hidden3)
        return output
    
class TwoHiddenMLP(nn.Module):
    def __init__(self, input_size=784, hidden1_size=352, hidden2_size=352, output_size=10, 
                 freeze=False, use_weight_clipping=True):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        # Add pruning attribute to avoid AttributeError
        self.pruning = False
        self.use_weight_clipping = use_weight_clipping
        self.percentile_clip_min = 10  # Bottom 10 percentile
        self.percentile_clip_max = 90  # Top 10 percentile

        self.input_to_hidden1 = nn.Linear(input_size, hidden1_size, bias=True)
        if freeze:
            self.register_buffer('hidden1_to_hidden2', torch.randn(hidden1_size, hidden2_size))
        else:
            self.hidden1_to_hidden2 = nn.Parameter(torch.randn(hidden1_size, hidden2_size))
        self.hidden2_to_output = nn.Linear(hidden2_size, output_size, bias=True)
        self.relu = nn.ReLU()
        
        # Initialize with clipping if enabled - only applied once during initialization
        if self.use_weight_clipping:
            self.clip_weights()
            print("Applied weight clipping during initialization only")

    def clip_weights(self):
        """
        Clip weights to lie between percentiles of their distribution.
        This is applied to hidden1_to_hidden2 weights only.
        """
        if not hasattr(self, 'hidden1_to_hidden2') or not isinstance(self.hidden1_to_hidden2, nn.Parameter):
            return
            
        with torch.no_grad():
            # Flatten weights for percentile calculation
            flat_weights = self.hidden1_to_hidden2.view(-1)
            
            # Calculate percentiles
            min_val = torch.quantile(flat_weights, self.percentile_clip_min / 100.0)
            max_val = torch.quantile(flat_weights, self.percentile_clip_max / 100.0)
            
            # Clip weights to the range [min_val, max_val]
            self.hidden1_to_hidden2.data.clamp_(min_val, max_val)
            
            print(f"Clipped weights to range [{min_val.item():.4f}, {max_val.item():.4f}] "
                  f"({self.percentile_clip_min}% to {self.percentile_clip_max}%)")

    def forward(self, x):
        x = x.view(x.size(0), -1)

        hidden1 = self.relu(self.input_to_hidden1(x))
        
        # No weight clipping during forward pass - removed from training process
        hidden2 = self.relu(torch.matmul(hidden1, self.hidden1_to_hidden2))
        output = self.hidden2_to_output(hidden2)
        return output   