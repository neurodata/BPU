import numpy as np

def normalize_matrix(W, mode=None):
    """
    Normalize a matrix using specified mode.
    
    Args:
        W (np.ndarray): Input matrix to normalize
        mode (str): Normalization mode ('minmax', 'clip', or None)
    
    Returns:
        np.ndarray: Normalized matrix
    """
    if mode is None:
        return W
        
    if mode == 'minmax':
        return (W - np.min(W)) / (np.max(W) - np.min(W))
    elif mode == 'clip':
        # Use 10th and 90th percentiles as clip range
        min_val = np.percentile(W, 10)
        max_val = np.percentile(W, 90)
        W = np.clip(W, min_val, max_val)
        return (W - min_val) / (max_val - min_val)  # Normalize to [0,1]
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

def get_weight_matrix(base, mode):
    """
    Generate weight matrices based on different initialization modes.
    
    Args:
        base (np.ndarray): Base matrix to use as reference
        mode (str): Initialization mode. Options:
            - 'random': He initialization for ReLU
            - 'droso': Use base matrix as is
            - 'permuted': Randomly permute nonzero values
            - 'sparsity_matched': Random sparse matrix with same sparsity
            - 'row_permuted': Randomly permute rows of base matrix
            - 'col_permuted': Randomly permute columns of base matrix
            - 'eigenvalue_matched': Random matrix matching eigenvalue distribution
            - 'eigenvalue_permuted': Matrix with same eigenvalue distribution by permuting/discarding values
    
    Returns:
        np.ndarray: Generated weight matrix
    """
    if mode == 'random':
        # use He Initialization for ReLU
        arr_np = (np.random.randn(*base.shape) / np.sqrt(base.shape[0])).astype(np.float32)
        return arr_np
    
    elif mode == 'droso':
        return base
    
    elif mode == 'permuted':
        nonzero_vals = base[base != 0].astype(np.float32)
        np.random.shuffle(nonzero_vals)
        
        non_zero_count = len(nonzero_vals)
        idx = np.random.choice(base.size, non_zero_count, replace=False)
        arr_np = np.zeros_like(base, dtype=np.float32)
        
        arr_np_flat = arr_np.flatten()
        arr_np_flat[idx] = nonzero_vals
        arr_np = arr_np_flat.reshape(base.shape)
        
        return arr_np
        
    elif mode == 'sparsity_matched':
        non_zero = np.count_nonzero(base)
        mask = np.zeros(base.shape, dtype=np.float32)
        idx = np.random.permutation(mask.size)[:non_zero]
        mask.flat[idx] = 1
        scaling_factor = np.sqrt(non_zero / base.size)  # normalization factor
        arr_np = (np.random.randn(*base.shape) * scaling_factor).astype(np.float32) * mask
        return arr_np
        
    elif mode == 'row_permuted':
        # Randomly permute rows while preserving column structure
        row_indices = np.random.permutation(base.shape[0])
        return base[row_indices, :]
        
    elif mode == 'col_permuted':
        # Randomly permute columns while preserving row structure
        col_indices = np.random.permutation(base.shape[1])
        return base[:, col_indices]
        
    elif mode == 'eigenvalue_matched':
        # Get eigenvalue distribution of base matrix
        eigenvalues = np.linalg.eigvals(base)
        magnitudes = np.abs(eigenvalues)
        
        # Create random matrix with same shape
        random_matrix = np.random.randn(*base.shape)
        
        # Get its eigenvalues
        random_eigenvalues = np.linalg.eigvals(random_matrix)
        random_magnitudes = np.abs(random_eigenvalues)
        
        # Scale random eigenvalues to match base eigenvalue magnitudes
        scaling_factor = np.mean(magnitudes) / np.mean(random_magnitudes)
        scaled_eigenvalues = random_eigenvalues * scaling_factor
        
        # Reconstruct matrix with scaled eigenvalues
        # Note: This is an approximation since we can't perfectly reconstruct
        # the original matrix from just eigenvalues
        U, _ = np.linalg.qr(random_matrix)
        D = np.diag(scaled_eigenvalues)
        return U @ D @ U.T
        
    elif mode == 'eigenvalue_permuted':
        # Get eigenvalue distribution of base matrix
        eigenvalues = np.linalg.eigvals(base)
        magnitudes = np.abs(eigenvalues)
        
        # Get the rank of the base matrix
        rank = np.linalg.matrix_rank(base)
        
        # Sort eigenvalues by magnitude
        sorted_indices = np.argsort(magnitudes)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        
        # Keep only the top 'rank' eigenvalues
        kept_eigenvalues = sorted_eigenvalues[:rank]
        
        # Create a diagonal matrix with the kept eigenvalues
        D = np.zeros_like(base, dtype=np.complex128)
        D[:rank, :rank] = np.diag(kept_eigenvalues)
        
        # Generate random orthogonal matrix
        Q = np.random.randn(*base.shape)
        Q, _ = np.linalg.qr(Q)
        
        # Reconstruct matrix
        reconstructed = Q @ D @ Q.T
        
        # Scale to match the magnitude distribution of the original matrix
        orig_magnitude = np.mean(np.abs(base))
        reconstructed_magnitude = np.mean(np.abs(reconstructed))
        scaling_factor = orig_magnitude / reconstructed_magnitude
        
        return (reconstructed * scaling_factor).astype(np.float32)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
