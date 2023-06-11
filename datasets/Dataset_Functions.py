
import pandas as pd
from tqdm import tqdm

import torch
from torch.sparse import FloatTensor



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






