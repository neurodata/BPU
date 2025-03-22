import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    return test_dataset

def classify_images(model, dataset, device='cpu'):
    """Classify images and return predictions along with ground truth."""
    results = []
    for img, label in dataset:
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
        predicted = output.argmax(dim=1).item()
        results.append({
            'image': img.squeeze().cpu().numpy(),
            'true_label': label,
            'predicted_label': predicted,
            'correct': predicted == label
        })
    return results

def create_visualization(results):
    """Create a 10x10 grid visualization of misclassified MNIST digits.
    Each column represents a predicted digit (0-9) that was incorrect."""
    # Create 10 rows and 10 columns
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    
    # Group misclassified results by predicted label
    by_predicted = {i: [] for i in range(10)}
    for r in results:
        if not r['correct']:  # Only include misclassified examples
            by_predicted[r['predicted_label']].append(r)
    
    for pred_digit in range(10):
        misclassified = by_predicted[pred_digit]
        axes[0, pred_digit].set_title(f"{pred_digit}", fontsize=20, pad=10)
        
        if len(misclassified) > 10:
            selected_examples = list(np.random.choice(misclassified, 10, replace=False))
        else:
            selected_examples = misclassified[:10]
        
        for row in range(10):
            ax = axes[row, pred_digit]
            
            if row < len(selected_examples):
                example = selected_examples[row]
                ax.imshow(example['image'], cmap='gray')
            else:
                ax.imshow(np.ones((28, 28)) * 0.5, cmap='gray')
            
            ax.axis('off')
    
    fig.suptitle("Columns show digits predicted by model (all examples are misclassified)", 
                fontsize=14, y=0.98)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig