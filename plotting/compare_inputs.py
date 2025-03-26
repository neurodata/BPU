import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = "../results"
SELECTED_SAMPLE = "5%"  # Assuming this matches plot_metrics.ipynb

# Define colors for sensory types - these are the base labels
SENSORY_COLORS = {
    "Visual": "#E41A1C",            # Red
    "Olfactory": "#377EB8",         # Blue
    "Gut": "#4DAF4A",               # Green
    "Respiratory": "#984EA3",       # Purple
    "Gustatory-External": "#FF7F00", # Orange
    "All Sensory": "#000000"        # Black
}

def load_sensory_comparison_results():
    """
    Load the results for different sensory input types.
    """
    data = {}
    
    # Map of experiment names to sensory types for better labeling
    # This maps to the exact experiment names in the result files
    experiment_to_type = {
        "Unlearnable_Visual_DPU": "Visual",
        "Unlearnable_Olfactory_DPU": "Olfactory",
        "Unlearnable_Gut_DPU": "Gut",
        "Unlearnable_Respiratory_DPU": "Respiratory",
        "Unlearnable_Gustatory_DPU": "Gustatory-External",
        "Unlearnable_All_Sensory_DPU": "All Sensory"
    }
    
    # Count neurons for each sensory type to enhance the labels
    neuron_counts = {}
    
    for fname in os.listdir(RESULTS_DIR):
        for exp_name, sensory_type in experiment_to_type.items():
            if fname.startswith(exp_name):
                with open(os.path.join(RESULTS_DIR, fname), "rb") as f:
                    results = pickle.load(f)
                    
                    # Store the results using the base sensory type as key
                    data.setdefault(sensory_type, []).append(results)
                    
                    # Extract neuron count if available
                    if isinstance(results, list) and len(results) > 0:
                        first_trial = results[0]
                        if 'sensory_dim' in first_trial:
                            neuron_counts[sensory_type] = first_trial['sensory_dim']
    
    # Now enhance the labels with neuron counts if available
    enhanced_data = {}
    for sensory_type, results in data.items():
        if sensory_type in neuron_counts:
            enhanced_label = f"{sensory_type} ({neuron_counts[sensory_type]} neurons)"
        else:
            enhanced_label = sensory_type
        enhanced_data[enhanced_label] = results
    
    return enhanced_data

def plot_sensory_comparison():
    """
    Plot performance comparison across different sensory input types.
    """
    plt.figure(figsize=(10, 8))
    
    data = load_sensory_comparison_results()
    
    x_max = 0
    
    # Plot each sensory type
    for i, (sensory_label, acc_lists) in enumerate(sorted(data.items())):
        # Extract the base sensory type from the enhanced label (e.g., "Visual (123 neurons)" -> "Visual")
        base_sensory_type = sensory_label.split(" (")[0] 
        
        # Convert list of accuracy lists to numpy array
        acc_matrix = np.array([acc_list['test_acc'] if 'test_acc' in acc_list else acc_list['epoch_test_acc'] for acc_list in acc_lists])
        epochs = acc_matrix.shape[1]
        
        # Calculate mean and standard deviation across trials
        mean = np.mean(acc_matrix, axis=0)
        std = np.std(acc_matrix, axis=0, ddof=1)
        
        # Consistent with other plots, for fewshot_300 results
        target_samples = {"1%": 60, "2%": 120, "5%": 300, "10%": 600, "100%": 0.1}[SELECTED_SAMPLE]
        x = np.arange(epochs) * target_samples * 10
        x_max = max(x_max, x[-1])
        
        # Get color based on the base sensory type
        color = SENSORY_COLORS.get(base_sensory_type)
        if color is None:
            # Fallback to default colors if not found
            color = plt.cm.tab10(i % 10)
        
        # Plot mean line
        plt.plot(
            x, mean,
            color=color,
            label=sensory_label,
            linewidth=2.5
        )
        
        # Plot standard deviation as shaded area
        plt.fill_between(
            x, mean - std, mean + std,
            color=color,
            alpha=0.15
        )
    
    plt.xlabel("Training Samples" if SELECTED_SAMPLE != "100%" else "Epochs", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xlim(0, x_max*1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    
    plt.legend(loc='lower right', fontsize=12)
    plt.title("Performance Comparison Across Different Sensory Input Types", fontsize=16)
    
    # Create figures directory if it doesn't exist
    os.makedirs("./figures", exist_ok=True)
    
    plt.savefig("./figures/sensory_comparison.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_sensory_comparison() 