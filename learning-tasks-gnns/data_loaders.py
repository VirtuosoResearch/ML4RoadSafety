import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected, coalesce, negative_sampling

def load_monthly_data(data, data_dir = "./data", state_name = "MA", num_negative_edges = 100000, year=2022, month=1):
    '''
    Return:
        - node features of the month
        - edge features of the month
        - accidents of the month: every accident labeled with (year_month, node_1_idx, node_2_idx, acc_count)
        - negative edges 
    '''
    # Load accidents
    accidents = pd.read_csv(os.path.join(data_dir, f"{state_name}/accidents_monthly.csv"))

    monthly_accidents = accidents[accidents["year"] == year]
    monthly_accidents = monthly_accidents[monthly_accidents["month"] == month]
    monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count", "year", "month"]].values

    pos_edges = torch.Tensor(monthly_accidents[:, :2])
    pos_edge_weights = torch.Tensor(monthly_accidents[:, 2])
    pos_edges, pos_edge_weights = coalesce(pos_edges.T, pos_edge_weights)
    pos_edges = pos_edges.type(torch.int64).T

    # sample negative edges from rest of edges in the road network
    all_edges = data.edge_index.cpu().T.numpy()
    neg_mask = np.logical_not(np.isin(all_edges, pos_edges.numpy()).all(axis=1))
    neg_edges = all_edges[neg_mask]
    rng = np.random.default_rng(year * 12 + month)
    neg_edges = neg_edges[rng.choice(neg_edges.shape[0], num_negative_edges, replace=False)]
    neg_edges = torch.Tensor(neg_edges).type(torch.int64)

    # load the node features of the month
    node_features = pd.read_csv(os.path.join(data_dir, f"{state_name}/Nodes/node_features_{year}_{month}.csv"))
    node_features = node_features[["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]]
    node_features = node_features.fillna(node_features.mean(axis=0))
    node_features = node_features.fillna(0)
    node_features = torch.Tensor(node_features.values)

    # load the edge features 
    edge_feature_dir = os.path.join(data_dir, f"{state_name}/Edges/edge_features_{year}_1.pt")
    if os.path.exists(edge_feature_dir):
        edge_feature_dict = torch.load(edge_feature_dir)

        # for key in ['AADT']: edge_features = torch.stack(edge_features, dim=1)
        column_values = edge_feature_dict['AADT'].coalesce().values()
        column_values_mean = column_values[~torch.isnan(column_values)].mean()
        column_values[torch.isnan(column_values)] = 0 if torch.isnan(column_values_mean) else column_values_mean
        edge_features = column_values.view(-1, 1)
    else:
        edge_features = torch.zeros(data.edge_index.shape[1], 1)

    return pos_edges, pos_edge_weights, neg_edges, node_features, edge_features


def load_static_edge_features(data_dir = "./data", state_name = "MA"):
    edge_feature_dict = torch.load(os.path.join(data_dir, f"{state_name}/Edges/edge_features.pt"))

    edge_lengths = edge_feature_dict['length'].coalesce().values()
    length_mean = edge_lengths[~torch.isnan(edge_lengths)].mean()
    edge_lengths[torch.isnan(edge_lengths)] = length_mean
    normalized_edge_lengths = (edge_lengths - length_mean) / torch.std(edge_lengths)

    edge_features = []
    for key in ['oneway', 'access_ramp', 'bus_stop', 'crossing', 'disused', 'elevator', 'escape', 'living_street', 'motorway', 'motorway_link', 'primary', 'primary_link', 'residential', 'rest_area', 'road', 'secondary', 'secondary_link', 'stairs', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified', 'unsurfaced']:
        if key not in edge_feature_dict: 
            column_values = torch.zeros(edge_lengths.shape)
        else:
            column_values = edge_feature_dict[key].coalesce().values()
            column_values[torch.isnan(column_values)] = 0
        edge_features.append(column_values)
    edge_features.append(normalized_edge_lengths)
    edge_features = torch.stack(edge_features, dim=1)
    return edge_features

def load_static_network(data_dir = "./data", state_name = "MA", 
                        feature_type = "verse", feature_name = "MA_ppr_128.npy"):
    # Load adjacency matrix
    adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
    edge_index = adj.coalesce().indices().long()

    data = Data(edge_index=edge_index)
    # data = T.ToUndirected()(data)

    # load node embeddings
    embedding_dir = os.path.join("./embeddings/", f"{feature_type}/{feature_name}")
    if os.path.exists(embedding_dir):
        node_embeddings = np.load(embedding_dir)
        data.x = torch.Tensor(node_embeddings)

    return data

def generate_accident_edges(accidents, years = [2002, 2013], months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    edges = []
    edge_weights = []
    edge_years = []
    edge_months = []

    for year in years:
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
                                train_years = list(range(2002, 2013)), train_months = list(range(1, 13)),
                                valid_years = list(range(2013, 2018)), valid_months = list(range(1, 13)),
                                test_years = list(range(2018, 2023)), test_months = list(range(1, 13)),
                                feature_type = "verse", feature_name = "MA_ppr_128.npy"):
    '''
    Return:
        - data: with the full traffic networks
        - splits: training edges, validation edges, validation negative edges, test edges, test negative edges
    '''
    # Load adjacency matrix
    adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
    edge_index = adj.coalesce().indices()

    data = Data(edge_index=edge_index)
    # data = T.ToUndirected()(data)
    num_nodes = data.num_nodes

    # Load accidents: split training, validation and test
    accidents = pd.read_csv(os.path.join(data_dir, f"{state_name}/accidents_monthly.csv"))

    training_edges, training_edge_weights = generate_accident_edges(accidents, years=train_years, months=train_months)
    valid_edges, valid_edge_weights = generate_accident_edges(accidents, years=valid_years, months=valid_months)
    test_edges, test_edge_weights = generate_accident_edges(accidents, years=test_years, months=test_months)

    # sample negative edges
    neg_valid_edges = negative_sampling(valid_edges, num_nodes=num_nodes, num_neg_samples=num_negative_edges)
    neg_test_edges  = negative_sampling(test_edges, num_nodes=num_nodes, num_neg_samples=num_negative_edges)

    split_edge = {
        "train": {"edge": training_edges.T, "weight": training_edge_weights},
        "valid": {"edge": valid_edges.T, "weight": valid_edge_weights, "edge_neg": neg_valid_edges.T},
        "test": {"edge": test_edges.T, "weight": test_edge_weights, "edge_neg": neg_test_edges.T}
    }

    embedding_dir = os.path.join("./embeddings/", f"{feature_type}/{feature_name}")
    node_embeddings = np.load(embedding_dir)
    data.x = torch.Tensor(node_embeddings)

    return data, split_edge
