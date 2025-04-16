
import pandas as pd
import numpy as np
import os
"""
Veryfied same neuron id list, but different order
"""
conn_file_list = [
                  'data/ad_connectivity_matrix.csv',
                  'data/signed_connectivity_matrix.csv',
                  'paper_data/Supplementary-Data-S1/aa_connectivity_matrix.csv',
                  'paper_data/Supplementary-Data-S1/ad_connectivity_matrix.csv',
                  'paper_data/Supplementary-Data-S1/da_connectivity_matrix.csv',
                  'paper_data/Supplementary-Data-S1/dd_connectivity_matrix.csv',
                  'paper_data/Supplementary-Data-S1/all-all_connectivity_matrix.csv',
                  ]
neuron_id = None
for file_path in conn_file_list:
    df = pd.read_csv(os.path.join('..',file_path))
    df.set_index(df.columns[0], inplace=True)
    cur_neuron_id = df.index.to_numpy()
    cur_neuron_id.sort()
    # print((cur_neuron_id).shape)
    if neuron_id is None:
        neuron_id = cur_neuron_id
    elif not np.all(neuron_id == cur_neuron_id):
        raise ValueError('Neuron IDs do not match')
    a = 1