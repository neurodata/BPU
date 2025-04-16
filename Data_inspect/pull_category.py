import pandas as pd
import numpy as np
import os

def get_sorted_neuron_id():
    conn_file_path = 'data/droso_data/signed_connectivity_matrix.csv'
    df = pd.read_csv(os.path.join('..',conn_file_path))
    df.set_index(df.columns[0], inplace=True)
    neuron_id = df.index.to_numpy().astype(int)
    neuron_id.sort()
    return neuron_id

output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}

neuron_id = get_sorted_neuron_id()
neuron_df = pd.DataFrame({'ID': neuron_id, 'Category': ['None'] * len(neuron_id),'sub_Category': [''] * len(neuron_id)})
#load s3 and s4
s3_df = pd.read_csv('../data/droso_data/paper_data/science.add9330_data_s3.csv')
s3_df.pop('axonIO_ratio')
s4_df = pd.read_csv('../data/droso_data/paper_data/science.add9330_data_s4.csv')
s4_df.pop('dendritic_output-input_ratio')
info_df = pd.concat([s3_df, s4_df], ignore_index=True)
info_df['celltype'] = ( # clean up so it match the naming in s2
    info_df['celltype']
    .str.replace(r's(?=-)', '', regex=True)  # removes an "s" before a hyphen
    .str.replace(r's$', '', regex=True)      # removes an "s" at the end of the string
)
info_df['skid'] = info_df['skid'].astype(int)
info_df['sub_type'] = [''] * len(info_df)
# cell_type_list1 = np.unique(info_df['celltype'].to_numpy())

# load s2
s2_df = pd.read_csv('../data/droso_data/paper_data/science.add9330_data_s2.csv')
# cell_type_list = np.unique(s2_df['celltype'].to_numpy())
for _, row in s2_df.iterrows():
    for col_name in ['left_id', 'right_id']:
        if row[col_name] != 'no pair':
            if row['celltype'] == 'sensory':
                info_df.loc[len(info_df)] = [int(row[col_name]), row['celltype'],row['additional_annotations']]
            else:
                info_df.loc[len(info_df)] = [int(row[col_name]), row['celltype'],'']


for idx, row in info_df.iterrows():
    cur_id,cur_category,cur_sub_category = row['skid'], row['celltype'],row['sub_type']
    selected = neuron_df.loc[neuron_df['ID'] == int(cur_id)]
    try:
        selected['Category'].iloc[0]
    except:
        print(idx,cur_id,cur_category,cur_sub_category)
        continue

    if selected['Category'].iloc[0] == 'None':
        neuron_df.loc[neuron_df['ID'] == cur_id, 'Category'] = cur_category
        neuron_df.loc[neuron_df['ID'] == cur_id, 'sub_Category'] = cur_sub_category
    elif selected['Category'].iloc[0] != cur_category:
        print(cur_id,cur_category,selected['Category'].iloc[0])
        neuron_df.loc[neuron_df['ID'] == cur_id, 'Category'] = 'Unclear' # TODO mayber label which area names -> theres only 4 neuron had this problem
        # raise ValueError('Mismatch in Category info')
    elif selected['sub_Category'].iloc[0] != cur_sub_category:
        print(cur_id, cur_sub_category, selected['sub_Category'].iloc[0])
        neuron_df.loc[neuron_df['ID'] == cur_id, 'sub_Category'] = 'Unclear'

i = 0
for idx, row in neuron_df.iterrows():
    if row['Category'] in output_types:
        i +=1
print(i)
temp_df = neuron_df[neuron_df['Category'] == 'KC']
neuron_df.to_pickle('neuron_category.pkl')
a = 1