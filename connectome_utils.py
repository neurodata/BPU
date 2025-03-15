import pandas as pd
import numpy as np

def load_drosophila_matrix(csv_path, signed=False):
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

    return W_norm

def load_connectivity_data(connectivity_path, annotation_path, rescale_factor=2e-2):
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
    adj_matrix = df_reindexed.values * rescale_factor

    num_S = len(valid_sensory_ids)
    return {
        'W': adj_matrix,
        'W_ss': adj_matrix[:num_S, :num_S],
        'W_sr': adj_matrix[:num_S, num_S:],
        'W_rs': adj_matrix[num_S:, :num_S],
        'W_rr': adj_matrix[num_S:, num_S:],
        'sensory_ids': valid_sensory_ids
    }

def load_sio_connectivity_data(connectivity_path, annotation_path, rescale_factor):
    """
    Load and process the connectivity matrix and neuron annotations, splitting neurons into
    Sensory, Internal, and Output groups, then return a dictionary of 9 connectivity sub-matrices
    (as NumPy arrays) plus the neuron ID lists.
    """

    df_annot = pd.read_csv(annotation_path)
    output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}
    sensory_ids = []
    output_ids = []

    for _, row in df_annot.iterrows():
        cell_type = row['celltype']
        additional_annotations = row['additional_annotations']
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                try:
                    neuron_id = int(id_str)
                except ValueError:
                    continue
                # Classify neuron
                if cell_type == 'sensory' and additional_annotations == 'visual':
                    sensory_ids.append(neuron_id)
                elif cell_type in output_types:
                    output_ids.append(neuron_id)

    sensory_ids = sorted(set(sensory_ids))
    output_ids = sorted(set(output_ids))

    print(f"Annotation file: Found {len(sensory_ids)} sensory neuron IDs")
    print(f"Annotation file: Found {len(output_ids)} output neuron IDs")

    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
    all_neuron_ids = sorted(df_conn.index.tolist())
    print(f"Connectivity matrix contains {len(all_neuron_ids)} neurons")

    valid_sensory_ids = [nid for nid in sensory_ids if nid in all_neuron_ids]
    valid_output_ids = [nid for nid in output_ids if nid in all_neuron_ids]

    # Define internal neurons as the rest ---
    valid_internal_ids = [
        nid for nid in all_neuron_ids
        if nid not in valid_sensory_ids and nid not in valid_output_ids
    ]

    print(f"After filtering, found {len(valid_sensory_ids)} sensory neurons in matrix")
    print(f"After filtering, found {len(valid_output_ids)} output neurons in matrix")
    print(f"Remaining {len(valid_internal_ids)} neurons classified as internal")

    adjacency = df_conn.values  # shape: [N, N]
    ordered_ids = df_conn.index.tolist() 
    id_to_idx = {nid: i for i, nid in enumerate(ordered_ids)}

    sensory_idx = [id_to_idx[nid] for nid in valid_sensory_ids]
    internal_idx = [id_to_idx[nid] for nid in valid_internal_ids]
    output_idx = [id_to_idx[nid] for nid in valid_output_ids]

    adjacency = adjacency * rescale_factor

    W_ss = adjacency[np.ix_(sensory_idx, sensory_idx)]
    W_sr = adjacency[np.ix_(sensory_idx, internal_idx)]
    W_so = adjacency[np.ix_(sensory_idx, output_idx)]

    W_rs = adjacency[np.ix_(internal_idx, sensory_idx)]
    W_rr = adjacency[np.ix_(internal_idx, internal_idx)]
    W_ro = adjacency[np.ix_(internal_idx, output_idx)]

    W_os = adjacency[np.ix_(output_idx, sensory_idx)]
    W_or = adjacency[np.ix_(output_idx, internal_idx)]
    W_oo = adjacency[np.ix_(output_idx, output_idx)]

    return {
        'W': adjacency,
        'W_ss': W_ss,
        'W_sr': W_sr,
        'W_so': W_so,
        'W_rs': W_rs,
        'W_rr': W_rr,
        'W_ro': W_ro,
        'W_or': W_or,
        'W_os': W_os,
        'W_oo': W_oo,
        'sensory_ids': valid_sensory_ids,
        'internal_ids': valid_internal_ids,
        'output_ids': valid_output_ids
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
