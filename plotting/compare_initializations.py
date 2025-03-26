import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_DIR = "../results/"
SELECTED_SAMPLE = "1%"  # option: "1%", "2%", "5%", "10%", "100%"

# Color map for different initialization methods
COLOR_MAP = {
    # Base models for comparison
    "Unlearnable_DPU": "#E41A1C",  # red
    "Learnable_RNN_No_Sparsity": "#377EB8",  # blue
    "Unlearnable_RNN_No_Sparsity": "#377EB8",  # blue
    
    # Permuted initialization
    "Learnable_RNN_Same_Sparsity": "#4DAF4A",  # green
    "Unlearnable_RNN_Same_Sparsity": "#4DAF4A",  # green
    
    # Row permuted initialization
    "Learnable_RNN_Row_Permuted": "#984EA3",  # purple
    "Unlearnable_RNN_Row_Permuted": "#984EA3",  # purple
    
    # Column permuted initialization
    "Learnable_RNN_Col_Permuted": "#FF7F00",  # orange
    "Unlearnable_RNN_Col_Permuted": "#FF7F00",  # orange
    
    # Eigenvalue matched initialization
    "Learnable_RNN_Eigenvalue_Matched": "#FFFF33",  # yellow
    "Unlearnable_RNN_Eigenvalue_Matched": "#FFFF33",  # yellow
    
    # Eigenvalue permuted initialization
    "Learnable_RNN_Eigenvalue_Permuted": "#A65628",  # brown
    "Unlearnable_RNN_Eigenvalue_Permuted": "#A65628",  # brown
}

# Line style map
LINE_STYLE = {
    # Learnable models use solid lines
    "Learnable_RNN_No_Sparsity": "-",
    "Learnable_RNN_Same_Sparsity": "-",
    "Learnable_RNN_Row_Permuted": "-",
    "Learnable_RNN_Col_Permuted": "-",
    "Learnable_RNN_Eigenvalue_Matched": "-",
    "Learnable_RNN_Eigenvalue_Permuted": "-",
    
    # Unlearnable models use dashed lines
    "Unlearnable_DPU": "--",
    "Unlearnable_RNN_No_Sparsity": "--",
    "Unlearnable_RNN_Same_Sparsity": "--",
    "Unlearnable_RNN_Row_Permuted": "--",
    "Unlearnable_RNN_Col_Permuted": "--",
    "Unlearnable_RNN_Eigenvalue_Matched": "--",
    "Unlearnable_RNN_Eigenvalue_Permuted": "--",
}

def load_results():
    target_samples = {"1%":60, "2%":120, "5%":300, "10%":600, "100%":None}[SELECTED_SAMPLE]
  
    data = {}
    pattern = re.compile(r"^(?P<exp>\w+?)(_fewshot_\d+)?(_trial\d+)?(\.signed)?\.pkl$")
  
    for fname in os.listdir(RESULTS_DIR):
        match = pattern.match(fname)
        if not match: continue
      
        exp_name = match.group("exp")
        if "_fewshot" in fname:
            samples = int(fname.split("_fewshot_")[1].split("_")[0])
            if samples != target_samples: continue
        else:
            if target_samples is not None: continue
      
        try:
            with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
                results = pickle.load(f)
                if 'epoch_test_acc' in results:
                    key = (exp_name, "learnable" if "Learnable" in exp_name else "unlearnable")
                    data.setdefault(key, []).append(results['epoch_test_acc'])
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
  
    return data

def create_custom_legend():
    legend_elements = [
        # Base models
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_DPU"], ls="--", label="DPU (U)"),
        Line2D([0], [0], color=COLOR_MAP["Learnable_RNN_No_Sparsity"], ls="-", label="Dense RNN (L)"),
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_RNN_No_Sparsity"], ls="--", label="Dense RNN (U)"),
        
        # Regular permutation
        Line2D([0], [0], color=COLOR_MAP["Learnable_RNN_Same_Sparsity"], ls="-", label="Permuted (L)"),
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_RNN_Same_Sparsity"], ls="--", label="Permuted (U)"),
        
        # Row permutation
        Line2D([0], [0], color=COLOR_MAP["Learnable_RNN_Row_Permuted"], ls="-", label="Row Permuted (L)"),
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_RNN_Row_Permuted"], ls="--", label="Row Permuted (U)"),
        
        # Column permutation
        Line2D([0], [0], color=COLOR_MAP["Learnable_RNN_Col_Permuted"], ls="-", label="Col Permuted (L)"),
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_RNN_Col_Permuted"], ls="--", label="Col Permuted (U)"),
        
        # Eigenvalue matched
        Line2D([0], [0], color=COLOR_MAP["Learnable_RNN_Eigenvalue_Matched"], ls="-", label="Eigenvalue Matched (L)"),
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_RNN_Eigenvalue_Matched"], ls="--", label="Eigenvalue Matched (U)"),
        
        # Eigenvalue permuted
        Line2D([0], [0], color=COLOR_MAP["Learnable_RNN_Eigenvalue_Permuted"], ls="-", label="Eigenvalue Permuted (L)"),
        Line2D([0], [0], color=COLOR_MAP["Unlearnable_RNN_Eigenvalue_Permuted"], ls="--", label="Eigenvalue Permuted (U)"),
    ]
    return legend_elements

def plot_results(data):
    plt.figure(figsize=(12, 9))
  
    x_max = 0
  
    for (exp_name, _), acc_list in data.items():
        if exp_name not in COLOR_MAP or exp_name not in LINE_STYLE:
            continue
            
        acc_matrix = np.array(acc_list)
        epochs = acc_matrix.shape[1]
      
        mean = np.mean(acc_matrix, axis=0)
        std = np.std(acc_matrix, axis=0, ddof=1)
      
        target_samples = {"1%":60, "2%":120, "5%":300,"10%":600, "100%":0.1}[SELECTED_SAMPLE]
        x = np.arange(epochs) * target_samples*10
        x_max = max(x_max, x[-1])
      
        plt.plot(
            x, mean,
            color=COLOR_MAP[exp_name],
            linestyle=LINE_STYLE[exp_name],
            linewidth=2.5,
            label=exp_name
        )
      
        plt.fill_between(
            x, mean - std, mean + std,
            color=COLOR_MAP[exp_name],
            alpha=0.15
        )
  
    plt.xlabel("Training Samples" if SELECTED_SAMPLE != "100%" else "Epochs", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xlim(0, x_max*1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(f"Performance Comparison of Different Initialization Methods ({SELECTED_SAMPLE} Training Data)", fontsize=16)
  
    # Create two plots: one for learnable, one for unlearnable models
    plot_by_category(data, "learnable")
    plot_by_category(data, "unlearnable")
    
    # Main plot with all models
    legend = plt.legend(
        handles=create_custom_legend(),
        loc="center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        frameon=False,
        fontsize=12
    )
  
    plt.savefig("./figures/init_comparison_all.pdf", bbox_inches="tight")
    plt.close()

def plot_by_category(data, category):
    """Create separate plots for learnable and unlearnable models"""
    plt.figure(figsize=(10, 8))
    
    x_max = 0
    
    for (exp_name, model_type), acc_list in data.items():
        # Skip if not in our color map or not the requested category
        if exp_name not in COLOR_MAP or exp_name not in LINE_STYLE or model_type != category:
            continue
            
        acc_matrix = np.array(acc_list)
        epochs = acc_matrix.shape[1]
        
        mean = np.mean(acc_matrix, axis=0)
        std = np.std(acc_matrix, axis=0, ddof=1)
        
        target_samples = {"1%":60, "2%":120, "5%":300,"10%":600, "100%":0.1}[SELECTED_SAMPLE]
        x = np.arange(epochs) * target_samples*10
        x_max = max(x_max, x[-1])
        
        # Use unique linestyle for better differentiation since all will be solid or dashed
        plt.plot(
            x, mean,
            color=COLOR_MAP[exp_name],
            linestyle="-" if "Learnable" in exp_name else "--",
            linewidth=2.5,
            label=exp_name.replace(f"{'Learnable' if 'Learnable' in exp_name else 'Unlearnable'}_RNN_", "")
        )
        
        plt.fill_between(
            x, mean - std, mean + std,
            color=COLOR_MAP[exp_name],
            alpha=0.15
        )
    
    plt.xlabel("Training Samples" if SELECTED_SAMPLE != "100%" else "Epochs", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xlim(0, x_max*1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(f"{category.capitalize()} Model Performance by Initialization ({SELECTED_SAMPLE} Training Data)", fontsize=16)
    
    # Create legend with simplified names
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    
    plt.savefig(f"./figures/init_comparison_{category}.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    experiment_data = load_results()
    plot_results(experiment_data) 