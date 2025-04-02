import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from tqdm import tqdm
from net import *
from connectome import *
from data_loader import DataLoader
from utils import get_weight_matrix

# Load config
with open("../config/config.yaml", "r") as f:
    config_data = yaml.safe_load(f)

# Global parameters
result_path = config_data.get("result_path", "../results")
signed = config_data.get("signed", True)
sio = config_data.get("sio", True)
num_trials = config_data.get("num_trials", 10)
num_epochs = config_data.get("num_epochs", 10)
batch_size = config_data.get("batch_size", 64)
learning_rate = config_data.get("learning_rate", 0.001)
experiments = config_data.get("experiments", {})

# Few-shot settings
fewshot_config = config_data.get("fewshot", {})
fewshot_enabled = fewshot_config.get("enabled", False)
fewshot_samples = fewshot_config.get("samples", 60)
fewshot_batch_size = fewshot_config.get("batch_size", 10)
if fewshot_enabled:
    fewshot_experiments = {}
    for exp_id, exp_config in experiments.items():
        cfg = exp_config.copy()
        cfg["fewshot"] = fewshot_samples
        cfg["fewshot_batch_size"] = fewshot_batch_size
        fewshot_experiments[f"{exp_id}_fewshot_{fewshot_samples}"] = cfg
    experiments = fewshot_experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(config):
    if config['type'] == 'basicrnn':
        conn = load_connectivity_data(
            connectivity_path=config_data["csv_paths"]["signed"],
            annotation_path=config_data["annotation_path"], 
            rescale_factor=config_data.get('rescale_factor', 4e-2), 
            sensory_type=config.get('sensory_type', 'all')
        )
        if sio:
            W_init = get_weight_matrix(conn['W'], config.get('init'))
        else:
            W_init = get_weight_matrix(conn['W_original'], config.get('init'))

        lora_config = config.get('lora', {})
        use_lora = lora_config.get('enabled', False)
        lora_rank = lora_config.get('rank', 8)
        lora_alpha = lora_config.get('alpha', 16)
        dropout_rate = config.get('dropout_rate', 0.2)

        return BasicRNN(
            W_init=W_init,
            input_dim=784,
            sensory_dim=conn['W_ss'].shape[0],
            internal_dim=conn['W_rr'].shape[0],
            output_dim=conn['W_oo'].shape[0],
            num_classes=10,
            sio=sio,
            trainable=config.get('trainable'),
            pruning=config.get('pruning'),
            target_nonzeros=np.count_nonzero(W_init),
            lambda_l1=config.get('lambda_l1'),
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout_rate=dropout_rate,
        )   
    elif config['type'] == 'threehiddenmlp':
        return ThreeHiddenMLP(784, 29, 147, 400, 10, config.get('freeze', False))
    elif config['type'] == 'twohiddenmlp':
        return TwoHiddenMLP(
            input_size=784, 
            hidden1_size=352, 
            hidden2_size=352, 
            output_size=10, 
            freeze=config.get('freeze', False),
            use_weight_clipping=config.get('use_weight_clipping', True)
        )
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, unit="batch", desc="Training")
    for data, target in pbar:
        data = data.squeeze(1).to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output = model(data)

        if hasattr(model, "pruning") and model.pruning:
            logits = model(data)
            ce_loss = F.cross_entropy(logits, target)
            l1_loss = model.lambda_l1 * model.get_l1_loss() if model.lambda_l1 is not None else 0
            loss = ce_loss + l1_loss
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
        train_acc = correct / total if total else 0
        pbar.set_postfix(loss=f"{loss.item():.4f}", train_acc=f"{train_acc:.2%}")
    
    if hasattr(model, "enforce_sparsity") and hasattr(model, "pruning") and model.pruning:
        print("enforce sparsity start, nonzeros: ", torch.count_nonzero(model.W).item())
        model.enforce_sparsity()
        print("enforce sparsity end, nonzeros: ", torch.count_nonzero(model.W).item())
    return total_loss / total, correct / total

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.squeeze(1).to(device)
            target = target.to(device)
            output = model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    acc = correct / total if total > 0 else 0
    return acc

def run_training_loop(model, config, full_train_set, test_loader, exp_id, trial_num, num_epochs, batch_size, fewshot_batch_size):
    results = {"epoch_train_loss": [],
               "epoch_train_acc": [],
               "epoch_test_acc": [],
               'submodules_nonzero': [],
               'similarity_dict': []}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    init_acc = evaluate(model, test_loader)
    results["epoch_test_acc"].append(init_acc)
    print(f"Trial {trial_num} | Epoch 0 | Test Acc: {init_acc:.2%}")
    
    # Create data loader instance
    data_loader = DataLoader(batch_size=batch_size)
    
    for epoch in range(num_epochs):
        # Get appropriate train loader based on configuration
        train_loader = data_loader.get_train_loader(
            full_train_set,
            epoch=epoch if "fewshot" in config else None,
            fewshot_samples=config.get("fewshot") if "fewshot" in config else None,
            fewshot_batch_size=config.get("fewshot_batch_size", fewshot_batch_size) if "fewshot" in config else None
        )

        epoch_loss, epoch_acc = train_epoch(model, optimizer, criterion, train_loader)
        results["epoch_train_loss"].append(epoch_loss)
        results["epoch_train_acc"].append(epoch_acc)

        test_acc = evaluate(model, test_loader)
        submodule_nonzero_dict = {}
        for name, submodule in model.named_children():
            sub_nonzero = 0
            for param in submodule.parameters(recurse=False):
                sub_nonzero += torch.count_nonzero(param).item()
            submodule_nonzero_dict[name] = sub_nonzero
        submodule_nonzero_dict['total'] = sum(torch.count_nonzero(p).item() for p in model.parameters())
        results['submodules_nonzero'].append(submodule_nonzero_dict)
        results["epoch_test_acc"].append(test_acc)

        print(f"submodule nonzero values: {submodule_nonzero_dict}")
        print(f"Trial {trial_num} | Epoch {epoch+1} | Test Acc: {test_acc:.2%}")

    if "fewshot" not in config:
        os.makedirs(os.path.join(result_path, "models"), exist_ok=True)
        model_filename = f"{exp_id}_trial{trial_num}.model.pkl"
        model_path = os.path.join(result_path, "models", model_filename)
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")

    return results

def save_results(exp_id, config, trial_num, results, signed):
    os.makedirs(result_path, exist_ok=True)
    filename = f"{exp_id}_trial{trial_num}"
    if "fewshot" in config:
        filename = f"{exp_id}_trial{trial_num}"
    if not sio:
        filename += ".whole"
    if signed:
        filename += ".signed"
    filename += ".pkl"
    with open(os.path.join(result_path, filename), "wb") as f:
        pickle.dump(results, f)

def train_experiment(exp_id, config, trial_num):
    print("========================================")
    print(f"Starting Experiment: {exp_id} Trial {trial_num}")
    print("Experiment configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("========================================\n")
    
    torch.manual_seed(trial_num)
    np.random.seed(trial_num)
    
    data_loader = DataLoader(batch_size=batch_size)
    full_train_set, test_loader = data_loader.load_data()
    
    model = initialize_model(config)
    model.to(device)
    results = run_training_loop(model, config, full_train_set, test_loader, exp_id, trial_num,
                              num_epochs, batch_size, fewshot_batch_size)
    save_results(exp_id, config, trial_num, results, signed)

if __name__ == "__main__":
    for exp_id, config in experiments.items():
        for trial_num in range(1, num_trials + 1):
            print(f"\n=== Training {exp_id} Trial {trial_num} ===")
            train_experiment(exp_id, config, trial_num)