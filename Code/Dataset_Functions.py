
import pandas as pd
from tqdm import tqdm

import torch
from torch.sparse import FloatTensor

from sklearn.preprocessing import OneHotEncoder






def one_hot_encode_features(df_edges, categorical_feats):

    # Create a OneHotEncoder for each categorical feature
    onehot_encoders = {}
    for f in categorical_feats:
        enc = OneHotEncoder()
        enc.fit(df_edges[[f]])
        onehot_encoders[f] = enc

    # Convert categorical feature values to one-hot encodings using OneHotEncoders
    for f in categorical_feats:
        enc = onehot_encoders[f].transform(df_edges[[f]]).toarray()
        df_enc = pd.DataFrame(enc, columns=[f + "_" + str(i) for i in range(enc.shape[1])])
        # drop the first one-hot encoded feature
        df_enc = df_enc.iloc[:, 1:]
        df_edges = pd.concat([df_edges, df_enc], axis=1)

    # Remove original categorical feature columns
    df_edges.drop(categorical_feats, axis=1, inplace=True)

    return df_edges



def create_node_features(df_nodes):

    # Create a dictionary to store the node features
    node_features = {}

    # Iterate over each feature in the DataFrame and create a sparse tensor for it
    for f in df_nodes.columns[1:]:
        # Convert the feature values to a tensor
        values = torch.FloatTensor(df_nodes[f].values)
        
        # Create a tensor of row indices for the sparse tensor
        row_indices = torch.arange(len(df_nodes))
        
        # Create a tensor of column indices for the sparse tensor
        col_indices = torch.zeros(len(df_nodes), dtype=torch.long)
        
        # Create a sparse tensor for the feature
        node_features[f] = torch.sparse.FloatTensor(torch.stack([row_indices, col_indices]), values, torch.Size([len(df_nodes), 1]))

    # Print the shape of the node features tensor for each feature
    # for f, tensor in node_features.items():
    #     print(f'{f}: {tensor.shape}')


    return node_features



def create_edge_features(df_nodes, df_edges):

    # Get the number of nodes in the graph
    num_nodes = len(df_nodes)

    # Initialize a dictionary to store the values, row indices, and column indices for the sparse tensor
    values_dict = {f: [] for f in df_edges.columns[2:]}
    row_indices_dict = {f: [] for f in df_edges.columns[2:]}
    col_indices_dict = {f: [] for f in df_edges.columns[2:]}

    # Iterate over each edge in the DataFrame and store its feature values in the dictionary
    for i, e in tqdm(df_edges.iterrows(), total=len(df_edges)):
        # Get the row and column indices for the sparse tensor
        row_idx = e['node_1']
        col_idx = e['node_2']

        # Get the indices of the row and column nodes in the node DataFrame
        row_node_idx = df_nodes[df_nodes['node_id'] == row_idx].index[0]
        col_node_idx = df_nodes[df_nodes['node_id'] == col_idx].index[0]

        # Store the feature values, row indices, and column indices in the dictionary
        for f in df_edges.columns[2:]:
            values_dict[f].append(e[f])
            row_indices_dict[f].append(row_node_idx)
            col_indices_dict[f].append(col_node_idx)

    # Create a sparse tensor for each edge feature
    edge_features = {}
    for f in tqdm(df_edges.columns[2:], total=len(df_edges.columns)-2):
        values = torch.FloatTensor(values_dict[f])
        row_indices = torch.LongTensor(row_indices_dict[f])
        col_indices = torch.LongTensor(col_indices_dict[f])
        edge_features[f] = torch.sparse.FloatTensor(
            torch.stack([row_indices, col_indices]),
            values,
            torch.Size([num_nodes, num_nodes])
        )

    return edge_features


def create_adjacency_matrix(df_nodes, df_edges):
    # Get the number of nodes in the graph
    num_nodes = len(df_nodes)
    
    # Create a dictionary to map node names to indices
    node_indices = {}
    for i, node in df_nodes.iterrows():
        node_indices[node['node_id']] = i

    # Initialize a dictionary to store the values, row indices, and column indices for the sparse tensor
    values_dict = {"weight": []}
    row_indices_dict = {"weight": []}
    col_indices_dict = {"weight": []}

    # Iterate over each edge in the DataFrame and store its weight in the dictionary
    for i, e in tqdm(df_edges.iterrows(), total=len(df_edges)):
        # Get the row and column indices for the sparse tensor
        row_idx = node_indices[e["node_1"]]
        col_idx = node_indices[e["node_2"]]

        # Store the weight, row indices, and column indices in the dictionary
        values_dict["weight"].append(e["length"])
        row_indices_dict["weight"].append(row_idx)
        col_indices_dict["weight"].append(col_idx)
        if(e["oneway"]==0):
            values_dict["weight"].append(e["length"])
            row_indices_dict["weight"].append(col_idx)
            col_indices_dict["weight"].append(row_idx)


    # Create a sparse tensor for the adjacency matrix
    values = torch.FloatTensor(values_dict["weight"])
    row_indices = torch.LongTensor(row_indices_dict["weight"])
    col_indices = torch.LongTensor(col_indices_dict["weight"])
    adj_matrix = torch.sparse.FloatTensor(
        torch.stack([row_indices, col_indices]),
        values,
        torch.Size([num_nodes, num_nodes]),
    )

    return adj_matrix






