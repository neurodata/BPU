import os

import numpy as np

from DPU_lib.droso_matrix.connectome import load_connectivity_info
from DPU_lib.droso_matrix.utils import get_weight_matrix
from DPU_lib.model.net import *


def initialize_model(config, config_data):
    data_setup = config_data.get('data')
    model_type = config.data.model_choice

    if data_setup['data_choice'] == 'chess_SV':
        num_out = config.data.num_return_buckets if data_setup['use_bucket'] else 2
    else:
        raise NotImplementedError("No other dataset is supported")

    conn = load_connectivity_info(
        cfg_data=config_data,
        input_type=config_data.get('input_type', 'all'),
        output_type=config_data.get('output_type', 'all'),
        sio=config_data.get('sio', True)
    )
    W_init = get_weight_matrix(conn['W'], config_data.get('init'))

    lora_config = config_data.get('lora', {})
    use_lora = lora_config.get('enabled', False)
    lora_rank = lora_config.get('rank', 8)
    lora_alpha = lora_config.get('alpha', 16)
    dropout_rate = config_data.get('dropout_rate', 0.2)

    if model_type == 'basicCNN':
        print(f"Use_RELU = {config_data.get('use_relu',False)}")
        return BasicCNN_new(
            W_init=W_init,
            sensory_idx=conn['sensory_idx'],
            KC_idx=conn['KC_idx'],
            internal_idx=conn['internal_idx'],
            output_idx=conn['output_idx'],
            num_out=num_out,
            trainable=config_data.get('trainable'),
            pruning=config_data.get('pruning'),
            target_nonzeros=np.count_nonzero(W_init),
            lambda_l1=config_data.get('lambda_l1'),
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            drop_out=config_data.get('drop_out', True),
            dropout_rate=dropout_rate,
            timesteps=config_data.get('timesteps'),
            filter_num = config_data.get('filter_num'),
            cumulate_output = config_data.get('cumulative', False),
            use_residual = config_data.get('residual',False),
            use_relu=config_data.get('use_relu',False),
            learnable_KC=config_data.get('learnable_KC', None),
        )

    if model_type == 'basicrnn':
        input_dim = (8 * 8 * 6 # piece location
                     + 4 # castling
                     + 16) # enpassant

        return BasicRNN_new(
            W_init=W_init,
            input_dim=input_dim,
            sensory_idx=conn['sensory_idx'],
            KC_idx = conn['KC_idx'],
            internal_idx=conn['internal_idx'],
            output_idx=conn['output_idx'],
            num_out=num_out,
            trainable=config_data.get('trainable'),
            pruning=config_data.get('pruning'),
            target_nonzeros=np.count_nonzero(W_init),
            lambda_l1=config_data.get('lambda_l1'),
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            drop_out = config_data.get('drop_out',True),
            dropout_rate=dropout_rate,
            timesteps=config_data.get('timesteps'),
            cumulate_output=config_data.get('cumulative', False),
            use_residual=config_data.get('residual', False),
            learnable_KC=config_data.get('learnable_KC', None),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
