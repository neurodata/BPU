import pandas as pd
import numpy as np

def load_drosophila_matrix(csv_path, apply_pruning=False, signed=False):
    """
    Load and process a Drosophila connectivity matrix.
    """
    W_df = pd.read_csv(csv_path, index_col=0, header=0)
    W = W_df.values.astype(np.float32)

    # Normalize depending on whether it's signed or unsigned
    if signed:
        max_abs = np.max(np.abs(W))
        W_norm = W / max_abs if max_abs != 0 else W
    else:
        W_min, W_max = W.min(), W.max()
        W_norm = (W - W_min) / (W_max - W_min + 1e-8)

    if apply_pruning:
        non_zero_count = np.count_nonzero(W)
        return W_norm, non_zero_count
    return W_norm

def load_connectivity_data(connectivity_path, annotation_path):
    """
    Load and preprocess connectivity matrix and annotation data for Drosophila.
    """
    # Load annotation file
    df_annot = pd.read_csv(annotation_path)

    # Extract IDs of visual sensory neurons
    mask = (df_annot['celltype'] == 'sensory') & (df_annot['additional_annotations'] == 'visual')
    sensory_visual_ids = []
    for _, row in df_annot[mask].iterrows():
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                sensory_visual_ids.append(int(id_str))

    # Remove duplicates and sort
    sensory_visual_ids = sorted(set(sensory_visual_ids))
    print(f"Found {len(sensory_visual_ids)} sensory-visual neuron IDs")

    # Load connectivity matrix
    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    # Filter valid IDs
    valid_sensory_ids = [nid for nid in sensory_visual_ids if nid in df_conn.index]
    other_ids = [nid for nid in df_conn.index if nid not in valid_sensory_ids]

    # Reindex and reorder
    df_reindexed = df_conn.loc[valid_sensory_ids + other_ids, valid_sensory_ids + other_ids]

    # Scale the matrix
    adj_matrix = df_reindexed.values * 2e-2

    num_S = len(valid_sensory_ids)
    return {
        'W_ss': adj_matrix[:num_S, :num_S],
        'W_sr': adj_matrix[:num_S, num_S:],
        'W_rs': adj_matrix[num_S:, :num_S],
        'W_rr': adj_matrix[num_S:, num_S:],
        'sensory_ids': valid_sensory_ids
    }

def load_sio_connectivity_data(connectivity_path, annotation_path):
    """
    Load and process connectivity matrix and neuron annotations (Sensory-Internal-Output).
    """
    # Load neuron annotations
    df_annot = pd.read_csv(annotation_path)

    # Extract IDs for sensory, internal, and output neurons
    output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}
    sensory_ids = []
    internal_brain_ids = []
    output_ids = []

    for _, row in df_annot.iterrows():
        cell_type = row['celltype']
        additional_annotations = row['additional_annotations']
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                neuron_id = int(id_str)
                if cell_type == 'sensory' and additional_annotations == 'visual':
                    sensory_ids.append(neuron_id)
                elif cell_type in output_types:
                    output_ids.append(neuron_id)
                else:
                    internal_brain_ids.append(neuron_id)

    # Remove duplicates and sort
    sensory_ids = sorted(set(sensory_ids))
    internal_brain_ids = sorted(set(internal_brain_ids))
    output_ids = sorted(set(output_ids))

    # Load connectivity matrix
    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    # Filter valid IDs
    valid_sensory_ids = [nid for nid in sensory_ids if nid in df_conn.index]
    valid_internal_ids = [nid for nid in internal_brain_ids if nid in df_conn.index]
    valid_output_ids = [nid for nid in output_ids if nid in df_conn.index]

    # Reorder
    all_neuron_ids = df_conn.index.tolist()
    valid_internal_ids = list(set(all_neuron_ids) - set(valid_sensory_ids) - set(valid_output_ids))
    valid_ids = valid_sensory_ids + valid_internal_ids + valid_output_ids
    df_reindexed = df_conn.loc[valid_ids, valid_ids]

    print(f"Found {len(valid_sensory_ids)} sensory neurons")
    print(f"Found {len(valid_internal_ids)} internal brain neurons")
    print(f"Found {len(valid_output_ids)} output neurons")

    # Scale
    adj_matrix = df_reindexed.values * 1e-2

    num_sensory = len(valid_sensory_ids)
    num_internal = len(valid_internal_ids)
    num_output = len(valid_output_ids)

    return {
        'W_ss': adj_matrix[:num_sensory, :num_sensory],
        'W_sr': adj_matrix[:num_sensory, num_sensory:num_sensory+num_internal],
        'W_so': adj_matrix[:num_sensory, num_sensory+num_internal:],

        'W_rs': adj_matrix[num_sensory:num_sensory+num_internal, :num_sensory],
        'W_rr': adj_matrix[num_sensory:num_sensory+num_internal, num_sensory:num_sensory+num_internal],
        'W_ro': adj_matrix[num_sensory:num_sensory+num_internal, num_sensory+num_internal:],

        'W_os': adj_matrix[num_sensory+num_internal:, :num_sensory],
        'W_or': adj_matrix[num_sensory+num_internal:, num_sensory:num_sensory+num_internal],
        'W_oo': adj_matrix[num_sensory+num_internal:, num_sensory+num_internal:],

        'sensory_ids': valid_sensory_ids,
        'internal_ids': valid_internal_ids,
        'output_ids': valid_output_ids
    }

def load_random_matrix(adj_matrix, num_S):
    """
    Load a random matrix for testing or fallback scenarios.
    """
    # Scale the matrix
    adj_matrix = adj_matrix * 1e-2

    # Mock sensory IDs
    sensory_ids = sorted(set(range(1, 30)))
    print(f"Found {len(sensory_ids)} sensory-visual neuron IDs (random)")

    return {
        'W_ss': adj_matrix[:num_S, :num_S],
        'W_sr': adj_matrix[:num_S, num_S:],
        'W_rs': adj_matrix[num_S:, :num_S],
        'W_rr': adj_matrix[num_S:, num_S:],
        'sensory_ids': sensory_ids
    }

def convert_unsigned_to_signed(sign_csv_path, adjacency_csv_path, out_csv_path):

    df = pd.read_csv(sign_csv_path)
    df["sign"] = df["sign"].apply(
        lambda x: "true" if str(x).strip().lower() == "true" else "false"
    )
    
    true_count = (df["sign"] == "true").sum()
    total_count = len(df)
    false_count = total_count - true_count
    true_pct = (true_count / total_count * 100) if total_count else 0
    false_pct = 100 - true_pct
    print(f"True sign count: {true_count} ({true_pct:.2f}%)")
    print(f"False sign count: {false_count} ({false_pct:.2f}%)")

    conn_matrix_df = pd.read_csv(adjacency_csv_path, index_col=0)
    conn_matrix_df.to_csv(out_csv_path)
    print(f"Saved signed connectivity matrix to: {out_csv_path}")
