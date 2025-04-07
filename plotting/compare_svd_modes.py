import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Get the project root directory
if '__file__' in globals():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Configuration
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SELECTED_SAMPLES = ["1%", "2%", "5%", "10%"]  # List of samples to plot

# Updated color map for experiments
COLOR_MAP = {
    "Unlearnable_RNN_Low_Rank": "#E41A1C",
    "Unlearnable_RNN_Random_Structure_Same_Spectrum": "#377EB8",
    "Unlearnable_RNN_Random_Singular_Values": "#4DAF4A",
    "Unlearnable_RNN_Identical_Singular_Values": "#984EA3",
    "Unlearnable_RNN_Random_Singular_Values_Random_Spectrum": "#FF7F00"
}

# Mapping from experiment names to display names
EXP_DISPLAY_NAMES = {
    "Unlearnable_RNN_Low_Rank": "Same λ Same v",
    "Unlearnable_RNN_Random_Structure_Same_Spectrum": "Same λ Random v",
    "Unlearnable_RNN_Random_Singular_Values": "Random λ Same v",
    "Unlearnable_RNN_Identical_Singular_Values": "Identical λ Same v",
    "Unlearnable_RNN_Random_Singular_Values_Random_Spectrum": "Random λ Random v"
}

def load_results(selected_sample):
    target_samples = {"1%":60, "2%":120, "5%":300, "10%":600}[selected_sample]
    data = {}
    pattern = re.compile(r"^(?P<exp>\w+?)(_fewshot_\d+)?(_trial\d+)?(\.whole)?(\.signed)?\.pkl$")
  
    for fname in os.listdir(RESULTS_DIR):
        match = pattern.match(fname)
        if not match: continue
      
        exp_name = match.group("exp")
        if exp_name not in COLOR_MAP:
            continue
            
        # Check if it's a whole matrix version
        is_whole = ".whole" in fname
        
        # Check if this is the right sample size
        if "_fewshot" in fname:
            samples = int(fname.split("_fewshot_")[1].split("_")[0])
            if samples != target_samples:
                continue
      
        try:
            with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
                results = pickle.load(f)
                if isinstance(results, dict):
                    # Create separate keys for SIO and Whole versions
                    exp_key = f"{exp_name}_whole" if is_whole else exp_name
                    
                    # Initialize the experiment data structure if it doesn't exist
                    if exp_key not in data:
                        data[exp_key] = {
                            'acc_lists': {},  # Dictionary to store accuracies for each variance
                            'is_whole': is_whole
                        }
                    
                    # Get trial number
                    trial_num = int(match.group("trial")[1:]) if match.group("trial") else 1
                    
                    # Store results for each variance threshold
                    for variance, trial_results in results.items():
                        if variance not in data[exp_key]['acc_lists']:
                            data[exp_key]['acc_lists'][variance] = []
                        # Get the final test accuracy (10th epoch)
                        final_acc = trial_results['epoch_test_acc'][-1]
                        data[exp_key]['acc_lists'][variance].append(final_acc)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
    
    # Convert lists of accuracies to numpy arrays and calculate statistics
    for exp_key in data:
        for variance in data[exp_key]['acc_lists']:
            acc_lists = data[exp_key]['acc_lists'][variance]
            if len(acc_lists) > 0:
                # Convert to numpy array
                acc_array = np.array(acc_lists)
                data[exp_key]['acc_lists'][variance] = acc_array
            else:
                print(f"Warning: No trials found for experiment {exp_key} with variance {variance}")
                continue
  
    return data

def plot_results(data, ax, selected_sample, global_y_max, panel_position):
    x_max = 0
    max_acc = 0
    
    # Get variance thresholds from the data
    variance_thresholds = sorted(next(iter(data.values()))['acc_lists'].keys())
    
    # First plot SIO versions, then whole matrix versions
    for exp_key, exp_data in data.items():
        # Extract base experiment name without _whole suffix
        exp_name = exp_key.replace('_whole', '')
        if exp_name not in COLOR_MAP:
            continue
            
        # Calculate mean accuracy for each variance threshold
        mean_accs = []
        std_accs = []
        ranks = []
        
        for variance in variance_thresholds:
            if variance not in exp_data['acc_lists']:
                continue
                
            acc_array = exp_data['acc_lists'][variance]
            if acc_array.size == 0:
                continue
                
            # Calculate mean and std of final accuracies
            mean_acc = np.mean(acc_array)
            std_acc = np.std(acc_array, ddof=1)
            
            # Calculate rank based on variance threshold
            # For variance=1.0, use the full rank
            if variance == 1.0:
                rank = min(exp_data['acc_lists'][variance].shape[0], exp_data['acc_lists'][variance].shape[1])
            else:
                # For other variances, calculate the rank needed to explain that variance
                U, s, _ = np.linalg.svd(exp_data['acc_lists'][variance])
                explained_variance = np.cumsum(s**2) / np.sum(s**2)
                rank = np.argmax(explained_variance >= variance) + 1
            
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)
            ranks.append(rank)
            
            max_acc = max(max_acc, mean_acc)
        
        # Plot with error bars
        x = ranks
        y = mean_accs
        yerr = std_accs
        
        # Use solid line for SIO and dashed line for whole matrix
        linestyle = '--' if exp_data['is_whole'] else '-'
        linewidth = 3.0
      
        ax.errorbar(
            x, y, yerr=yerr,
            color=COLOR_MAP[exp_name],
            linewidth=linewidth,
            linestyle=linestyle,
            marker='o',
            markersize=8,
            capsize=5
        )
    
    # Add max accuracy line
    rounded_max_acc = round(max_acc, 2)
    ax.axhline(y=rounded_max_acc, color='gray', linestyle='--', alpha=0.7)
    
    # Set y-axis ticks
    y_ticks = [0, rounded_max_acc, 1.0]
    y_labels = ['0', f'{rounded_max_acc}', '']
    ax.set_ylim(0, 1)  
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=36)
    
    # Set x-axis ticks
    ax.set_xlabel("Rank", fontsize=32)
    
    # Set axis labels only for appropriate panels
    if panel_position == 'A':  # Only first panel has y-axis label
        ax.set_ylabel("Test Accuracy", fontsize=32)
    else:
        ax.set_ylabel("")
    
    # Set title for all panels with panel letter integrated
    target_samples = {"1%":60, "2%":120, "5%":300, "10%":600}[selected_sample]
    ax.set_title(f"{panel_position}. {target_samples*10} samples per epoch", fontsize=36, pad=20)
    
    # Set grid
    ax.grid(True, axis='y', alpha=0.3)  # Only horizontal grid lines
    
    # Make tick marks thicker
    ax.tick_params(width=2, length=10)
    
    return max_acc

def create_custom_legend():
    legend_elements = []
    for exp_name, color in COLOR_MAP.items():
        display_name = EXP_DISPLAY_NAMES.get(exp_name, exp_name)
        # Add solid line for SIO version
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3.0, linestyle='-',
                  label=f"{display_name} (SIO)")
        )
        # Add dashed line for whole matrix version
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3.0, linestyle='--',
                  label=f"{display_name} (Whole)")
        )
    return legend_elements

if __name__ == "__main__":
    # Create a 1x4 subplot figure
    plt.rcParams['axes.linewidth'] = 2.0  # Make all axis lines thicker
    fig, axes = plt.subplots(1, 4, figsize=(30, 8))
    
    # First, collect all data and determine global max y-value
    all_data = {}
    global_max_acc = 0
    
    for sample in SELECTED_SAMPLES:
        experiment_data = load_results(sample)
        all_data[sample] = experiment_data
        
        # Find max accuracy across all panels
        for exp_key, exp_data in experiment_data.items():
            for variance, acc_array in exp_data['acc_lists'].items():
                if acc_array.size > 0:
                    mean_acc = np.mean(acc_array)
                    global_max_acc = max(global_max_acc, mean_acc)
    
    # Use a consistent y-axis scaling with a bit of padding
    global_y_max = max(1.0, global_max_acc * 1.1)
    
    # Plot each panel with consistent y-axis scaling
    panel_labels = ['A', 'B', 'C', 'D']
    for idx, sample in enumerate(SELECTED_SAMPLES):
        plot_results(all_data[sample], axes[idx], sample, global_y_max, panel_labels[idx])
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.25)  # Space between panels horizontally
    plt.tight_layout()
    
    # Add legend to the bottom of the figure
    legend = fig.legend(
        handles=create_custom_legend(),
        loc="center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        fontsize=24,
    )
    
    # Save figure
    figures_dir = os.path.join(PROJECT_ROOT, "plotting", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "svd_modes_performance.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "svd_modes_performance.png"), bbox_inches="tight", dpi=300)
    plt.close()
