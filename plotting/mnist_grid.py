import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml

# Get the project root directory (assuming this script is in plotting/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add src directory to Python path
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Load config
with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r") as f:
    config_data = yaml.safe_load(f)
result_path = os.path.join(PROJECT_ROOT, config_data.get("result_path", "results"))

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.load(model_path)
    model.eval()
    return model

def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(os.path.join(PROJECT_ROOT, "data"), train=False, download=True, transform=transform)
    return test_dataset

def classify_images(model, dataset, device='cpu', max_samples=None):
    """Classify images and return predictions along with ground truth.
    Optimized with optional max_samples and batch processing."""
    results = []
    
    # Create a data loader with batch size for faster processing
    batch_size = 64
    
    # Get the test loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Count how many samples we've processed
    sample_count = 0
    
    # Use a progress bar
    with tqdm(total=len(dataset) if max_samples is None else min(max_samples, len(dataset)), 
              desc="Classifying images") as pbar:
        for imgs, labels in test_loader:
            if max_samples is not None and sample_count >= max_samples:
                break
                
            # Process the whole batch at once
            imgs = imgs.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            
            predicted = outputs.argmax(dim=1)
            
            # Convert to CPU for further processing
            imgs_cpu = imgs.cpu().numpy()
            labels_cpu = labels.numpy()
            predicted_cpu = predicted.cpu().numpy()
            
            # Add results for this batch
            for i in range(len(imgs)):
                if max_samples is not None and sample_count >= max_samples:
                    break
                    
                results.append({
                    'image': imgs_cpu[i].squeeze(),
                    'true_label': labels_cpu[i],
                    'predicted_label': predicted_cpu[i],
                    'correct': predicted_cpu[i] == labels_cpu[i]
                })
                
                sample_count += 1
                pbar.update(1)
    
    return results

def create_visualization(results):
    """Create a 10x10 grid visualization of MNIST digits.
    Each column represents a digit (0-9) with correct examples and misclassified
    examples highlighted with red circles."""
    # Create 10 rows and 10 columns
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Added more vertical space
    
    # Calculate overall accuracy
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Calculate number of misclassified examples to show (rounding properly)
    error_rate = 1 - accuracy
    max_errors = round(100 * error_rate)  # Round instead of int() for proper rounding
    print(f"Showing {max_errors} misclassified examples based on {error_rate:.2%} error rate")

    # Group results by predicted label
    by_predicted = {i: {'correct': [], 'incorrect': []} for i in range(10)}
    for r in results:
        if r['correct']:
            by_predicted[r['predicted_label']]['correct'].append(r)
        else:
            by_predicted[r['predicted_label']]['incorrect'].append(r)
    
    # Select a limited number of misclassified examples based on accuracy
    all_incorrect = [r for r in results if not r['correct']]
    if len(all_incorrect) > max_errors:
        selected_incorrect = list(np.random.choice(all_incorrect, max_errors, replace=False))
    else:
        selected_incorrect = all_incorrect
    
    # Build dictionary of selected incorrect examples by predicted digit
    selected_by_predicted = {i: [] for i in range(10)}
    for r in selected_incorrect:
        selected_by_predicted[r['predicted_label']].append(r)
    
    # For each predicted digit column
    for col, pred_digit in enumerate(range(10)):
        # Make column label larger (similar to digit size)
        axes[0, col].text(0.5, 1.3, f"{pred_digit}", 
                         ha='center', va='center', 
                         fontsize=48,
                        #  fontweight='bold', 
                         transform=axes[0, col].transAxes)
        
        # Mix correct and incorrect examples, prioritizing incorrect
        incorrect_examples = selected_by_predicted[pred_digit]
        correct_examples = by_predicted[pred_digit]['correct']
        
        # Determine how many of each to show
        num_incorrect = len(incorrect_examples)
        num_correct = min(10 - num_incorrect, len(correct_examples))
        
        # Randomly select correct examples if needed
        if len(correct_examples) > num_correct:
            selected_correct = list(np.random.choice(correct_examples, num_correct, replace=False))
        else:
            selected_correct = correct_examples[:num_correct]
        
        # Combine examples
        all_examples = incorrect_examples + selected_correct
        np.random.shuffle(all_examples)  # Mix them up
        
        # Fill remaining slots with gray squares if needed
        while len(all_examples) < 10:
            all_examples.append(None)
        
        # Plot each example
        for row in range(10):
            ax = axes[row, col]
            example = all_examples[row]
            
            if example is not None:
                ax.imshow(example['image'], cmap='gray')
                
                # Add red circle around misclassified examples
                if not example['correct']:
                    rect = patches.Rectangle((0, 0), 28, 28, linewidth=5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
            else:
                # Empty gray square for empty slots
                ax.imshow(np.ones((28, 28)) * 0.5, cmap='gray')
            
            ax.axis('off')
    
    return fig

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(result_path, "models/Unlearnable_DPU_trial1.model.pkl")).to(device)
    # model = load_model(os.path.join(result_path, "models/DPU_LoRA_trial1.model.pkl")).to(device)
    # Load model and data
    dataset = get_mnist_data()

    # Classify images
    print("Classifying MNIST test images...")
    results = classify_images(model, dataset, device)

    # Create and save visualization
    print("Creating visualization...")
    fig = create_visualization(results)
    figures_dir = os.path.join(PROJECT_ROOT, "plotting", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, "mnist_visualization.pdf"), bbox_inches="tight")
    fig.show()