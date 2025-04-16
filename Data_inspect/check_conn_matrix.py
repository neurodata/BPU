import pandas as pd
import numpy as np
import itertools


BASE_CONN_DIR = '../data/droso_data/paper_data/Supplementary-Data-S1/'
connection_name = ['aa','ad','da','dd','all-all']
file_list = [f'{c}_connectivity_matrix.csv' for c in connection_name]

annotation_path = 'neuron_category.pkl'
df_annot = pd.read_pickle(annotation_path)

conn_dict = {}
for c in connection_name:
    file_name = f'{c}_connectivity_matrix.csv'
    file_path = f'{BASE_CONN_DIR}/{file_name}'
    df_conn = pd.read_csv(file_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
    all_neuron_ids = sorted(df_conn.index.tolist())
    df_conn_sio = df_conn.loc[all_neuron_ids, all_neuron_ids]
    conn_dict[c] = df_conn_sio.values
    print(c, np.sum(conn_dict[c] != 0), np.sum(conn_dict[c] != 0) / len(all_neuron_ids)/len(all_neuron_ids))



# Checking the relationships
# for signs in itertools.product([1, -1], repeat=4):
#     result = signs[0] * conn_dict['aa'] + signs[1] * conn_dict['ad']  + signs[2] * conn_dict['da']  + signs[3] * conn_dict['dd']
#
#     if np.allclose(result, conn_dict['all-all'] ):
#         print("Matching combination of signs found:")
#         print("aa * {} + ad * {} + da * {} + dd * {} = all_all".format(*signs))
#
# Matching combination of signs found:
# aa * 1 + ad * 1 + da * 1 + dd * 1 = all_all


a = 1
# load_sio_conn_zaq(connectivity_path, annotation_path, rescale_factor=4e-2, normalization='minmax',
#                                input_type='visual',
#                       output_type = 'output')
#
#
#     # df_annot = pd.read_csv(annotation_path)
#     output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}
#     # sensory_ids = []
#     # output_ids = []
#
#     df_category = pd.read_pickle(annotation_path)
#     if input_type in ['sensory', 'ascending']:
#         df_input = df_category[df_category['Category'] == input_type]
#     elif input_type != 'all':
#         df_input = df_category[df_category['sub_Category'] == input_type]
#     else:
#         df_input = df_category
#     input_ids = df_input['ID'].astype(int).tolist()
#
#     if output_type == 'output':
#         df_output = df_category[df_category['Category'].isin(output_types)]
#     elif output_type != 'all':
#         df_output = df_category[df_category['Category'] == output_type]
#     else:
#         df_output = df_category
#     output_ids = df_output['ID'].astype(int).tolist()
#
#     input_ids = sorted(set(input_ids))
#     output_ids = sorted(set(output_ids))
#
#     print(f"Annotation file: Found {len(input_ids)} {input_type} sensory neuron IDs")
#     print(f"Annotation file: Found {len(output_ids)} output neuron IDs")
#
#     df_conn = pd.read_csv(connectivity_path, index_col=0)
#     df_conn.index = df_conn.index.astype(int)
#     df_conn.columns = df_conn.columns.astype(int)
#     all_neuron_ids = sorted(df_conn.index.tolist())
#     print(f"Connectivity matrix contains {len(all_neuron_ids)} neurons")
#
#     valid_sensory_ids = [nid for nid in input_ids if nid in all_neuron_ids]
#     valid_output_ids = [nid for nid in output_ids if nid in all_neuron_ids]
#
#     # Define internal neurons as the rest ---
#     valid_internal_ids = [
#         nid for nid in all_neuron_ids
#         if nid not in valid_sensory_ids and nid not in valid_output_ids
#     ]
#
#     print(f"After filtering, found {len(valid_sensory_ids)} {input_type} sensory neurons in matrix")
#     print(f"After filtering, found {len(valid_output_ids)} output neurons in matrix")
#     print(f"Remaining {len(valid_internal_ids)} neurons classified as internal")
#
#     # Create the SIO-ordered adjacency matrix
#     ordered_ids = valid_sensory_ids + valid_internal_ids + valid_output_ids
#     df_conn_sio = df_conn.loc[ordered_ids, ordered_ids]
#     adjacency = df_conn_sio.values  # shape: [N, N]
#
#     # Apply normalization
#     adjacency = normalize_matrix(adjacency, mode=normalization)
#     adjacency = adjacency * rescale_factor
#
#     # Calculate indices for each group in the ISO-ordered matrix
#     num_sensory = len(valid_sensory_ids)
#     num_internal = len(valid_internal_ids)
#     num_output = len(valid_output_ids)
#
#     W_ss = adjacency[:num_sensory, :num_sensory]
#     W_sr = adjacency[:num_sensory, num_sensory:num_sensory + num_internal]
#     W_so = adjacency[:num_sensory, num_sensory + num_internal:]
#
#     W_rs = adjacency[num_sensory:num_sensory + num_internal, :num_sensory]
#     W_rr = adjacency[num_sensory:num_sensory + num_internal, num_sensory:num_sensory + num_internal]
#     W_ro = adjacency[num_sensory:num_sensory + num_internal, num_sensory + num_internal:]
#
#     W_os = adjacency[num_sensory + num_internal:, :num_sensory]
#     W_or = adjacency[num_sensory + num_internal:, num_sensory:num_sensory + num_internal]
#     W_oo = adjacency[num_sensory + num_internal:, num_sensory + num_internal:]
#
#     return {
#         'W': adjacency,  # Now in SIO order
#         'W_ss': W_ss,
#         'W_sr': W_sr,
#         'W_so': W_so,
#         'W_rs': W_rs,
#         'W_rr': W_rr,
#         'W_ro': W_ro,
#         'W_or': W_or,
#         'W_os': W_os,
#         'W_oo': W_oo,
#         'sensory_ids': valid_sensory_ids,
#         'internal_ids': valid_internal_ids,
#         'output_ids': valid_output_ids,
#         'input_type': input_type,
#         'output_type': output_type,
#     }