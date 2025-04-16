import os
import pickle
from pathlib import Path

import pandas as pd
import torch


def get_save_dir(config_data):
    if torch.backends.mps.is_available():
        base_dir = "results-mac-mar31"
    else:
        base_dir = "results"
    sub_folder_name = f"{config_data['neuron_config']['input_neuron']}-{config_data['neuron_config']['output_neuron']}"
    time_step_name = str(config_data['time_steps']) + 'Timesteps'
    save_dir = os.path.join(base_dir, sub_folder_name, time_step_name)
    return save_dir

def save_results(exp_id, config, trial_num, results, signed,config_data):
    save_dir = get_save_dir(config_data)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{exp_id}_trial{trial_num}"
    if "fewshot" in config:
        filename = f"{exp_id}_trial{trial_num}"
    if signed:
        filename += ".signed"
    filename += ".pkl"
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(results, f)

def get_input_output_list():
    info_df = pd.read_pickle('Data_inspect/neuron_category.pkl')
    sensory_df = info_df[info_df['Category'] == 'sensory']
    input_list = ['all', 'ascending', 'sensory'] + sorted(list(set(sensory_df['sub_Category'].to_list())))
    output_list = ['all', 'output', 'DN-SEZ', 'DN-VNC', 'RGN']
    return input_list, output_list

def get_ct(name):
    info_df = pd.read_pickle('Data_inspect/neuron_category.pkl')
    if name == 'all':
        return len(info_df)
    elif name == 'output':
        output_list = {'DN-SEZ', 'DN-VNC', 'RGN'}
        filterd = info_df[info_df['Category'].isin(output_list)]
        return len(filterd)
    else:
        filterd = info_df[info_df['Category'] == name]
        if filterd.empty:
            filterd = info_df[info_df['sub_Category'] == name]
        return len(filterd)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device.")

    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    return device

def get_project_base_dir():
    BASE_DIR = Path.cwd()
    if get_device() == torch.device("cuda"):
        return str(BASE_DIR)
    while BASE_DIR.name != 'Chess':
        BASE_DIR = BASE_DIR.parent
    return str(BASE_DIR)