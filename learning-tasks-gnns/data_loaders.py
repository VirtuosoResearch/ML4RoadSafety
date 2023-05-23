import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected, coalesce, negative_sampling

def generate_accident_edges(accidents, years = (2002, 2013), months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    edges = []
    edge_weights = []
    edge_years = []
    edge_months = []

    for year in range(years[0], years[1]):
        if len(months) == 12:
            monthly_accidents = accidents[accidents["year"] == year]
            monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count", "year", "month"]].values
            edges.append(monthly_accidents[:, :2])
            edge_weights.append(monthly_accidents[:, 2])
            # edge_years.append(monthly_accidents[:, 3])
            # edge_months.append(monthly_accidents[:, 4])
        else:
            for month in months:
                monthly_accidents = accidents[accidents["year"] == year][accidents["month"] == month]
                monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count", "year", "month"]].values
                edges.append(monthly_accidents[:, :2])
                edge_weights.append(monthly_accidents[:, 2])
                # edge_years.append(monthly_accidents[:, 3])
                # edge_months.append(monthly_accidents[:, 4])

    edges = torch.Tensor(np.concatenate(edges, axis=0))
    edge_weights = torch.Tensor(np.concatenate(edge_weights, axis=0))
    # edge_years = torch.Tensor(np.concatenate(edge_years, axis=0))
    # edge_months = torch.Tensor(np.concatenate(edge_months, axis=0))
    edges, edge_weights = coalesce(edges.T, edge_weights)
    edges = edges.type(torch.int64)
    return edges, edge_weights # 'year': edge_years, 'month': edge_months

def load_network_with_accidents(data_dir = "./data", state_name = "MA", num_negative_edges = 100000,
                                feature_type = "verse", feature_name = "MA_ppr_128.npy"):
    '''
    Return:
        - data: with the full traffic networks
            Use full network as adj
            Use only training positive edges as adj (TODO)
        - splits: training edges, validation edges, validation negative edges, test edges, test negative edges
    '''
    # Load adjacency matrix
    adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
    edge_index = adj.coalesce().indices()

    data = Data(edge_index=edge_index)
    data = T.ToUndirected()(data)
    num_nodes = data.num_nodes

    # Load accidents: split training, validation and test
    accidents = pd.read_csv(os.path.join(data_dir, f"{state_name}/accidents_monthly.csv"))

    training_edges, training_edge_weights = generate_accident_edges(accidents, years=(2002, 2013))
    valid_edges, valid_edge_weights = generate_accident_edges(accidents, years=(2013, 2018))
    test_edges, test_edge_weights = generate_accident_edges(accidents, years=(2018, 2023))

    # sample negative edges
    neg_valid_edges = negative_sampling(valid_edges, num_nodes=num_nodes, num_neg_samples=num_negative_edges)
    neg_test_edges = negative_sampling(test_edges, num_nodes=num_nodes, num_neg_samples=num_negative_edges)

    split_edge = {
        "train": {"edge": training_edges.T, "weight": training_edge_weights},
        "valid": {"edge": valid_edges.T, "weight": valid_edge_weights, "edge_neg": neg_valid_edges.T},
        "test": {"edge": test_edges.T, "weight": test_edge_weights, "edge_neg": neg_test_edges.T}
    }

    embedding_dir = os.path.join("./embeddings/", f"{feature_type}/{feature_name}")
    node_embeddings = np.load(embedding_dir)
    data.x = torch.Tensor(node_embeddings)

    return data, split_edge
