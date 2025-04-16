import torch

def data_trasnform(data,model_name,device):
    if model_name == 'BasicCNN' or model_name == 'BasicCNN_new':
        return data.to(device)
    elif model_name == 'BasicRNN' or model_name == 'BasicRNN_new':
        return data.to(device)
    else:
        raise NotImplementedError