import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected, coalesce, negative_sampling

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected, coalesce, negative_sampling
from pyDataverse.api import NativeApi, DataAccessApi
from pyDataverse.models import Dataverse
import zipfile

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("File successfully unzipped!")

class TrafficAccidentDataset:

    def __init__(self, state_name = "MA", data_dir = "./data",
                    node_feature_type = "node2vec", use_static_edge_features=True, use_dynamic_node_features=True, use_dynamic_edge_features=False, 
                    train_years=[], num_negative_edges=100000000):
        self.data_dir = data_dir
        self.state_name = state_name
        # download dataset
        self.download_dataset()

        self.node_feature_type = node_feature_type
        self.node_feature_name = f"{state_name}.npy" if node_feature_type == "centrality" else f"{state_name}_128.npy"

        self.use_static_edge_features = use_static_edge_features
        self.use_dynamic_node_features = use_dynamic_node_features
        self.use_dynamic_edge_features = use_dynamic_edge_features
        self.num_negative_edges = num_negative_edges
        
        self.data = self.load_static_network()
        if self.use_static_edge_features:
            self.data.edge_attr = self.load_static_edge_features()

        # collecting dynamic features normlization statistics
        self.node_feature_mean = None
        self.node_feature_std = None
        self.edge_feature_mean = None
        self.edge_feature_std = None
        if len(train_years) != 0 and (self.use_dynamic_node_features or self.use_dynamic_edge_features):
            self.node_feature_mean, self.node_feature_std, self.edge_feature_mean, self.edge_feature_std = self.compute_feature_mean_std(train_years)

    def download_dataset(self):
        if os.path.exists(os.path.join(self.data_dir, self.state_name)):
            print("Dataset already downloaded!")
            return

        base_url = 'https://dataverse.harvard.edu/'
        api_token= '9275ad13-e563-4d16-b5cd-b0ce5731fe73' # change api token here
        api = NativeApi(base_url, api_token)
        data_api = DataAccessApi(base_url, api_token)
        DOI = "doi:10.7910/DVN/V71K5R"
        dataset = api.get_dataset(DOI)

        files_list = dataset.json()['data']['latestVersion']['files']
        for file in files_list:
            filename = file["dataFile"]["filename"]
            file_id = file["dataFile"]["id"]
            if filename == f"{self.state_name}.zip":
                response = data_api.get_datafile(file_id)
                with open(os.path.join(self.data_dir, filename), "wb") as f:
                    f.write(response.content)
                    unzip_file(os.path.join(self.data_dir, filename), self.data_dir)
                    if not os.path.exists(os.path.join(self.data_dir, self.state_name)):
                        print("Moving files...")
                        os.system(f"mv ./data/ML4RoadSafety_graphs_{self.state_name}/{self.state_name} ./data")
                    print("File successfully downloaded!")

    def load_monthly_data(self, year=2022, month=1):
        '''
        Return:
            - node features of the month
            - edge features of the month
            - accidents of the month: every accident labeled with (year_month, node_1_idx, node_2_idx, acc_count)
            - negative edges 
        '''
        monthly_data = {}

        # Load accidents
        accident_dir = f"{self.state_name}/accidents_monthly.csv"
        if not os.path.exists(os.path.join(self.data_dir, accident_dir)):
            new_data = self.data.clone()
            monthly_data['data'] = new_data
            monthly_data['x'] = new_data.x
            monthly_data['edge_index'] = new_data.edge_index
            monthly_data['edge_attr'] = new_data.edge_attr
            monthly_data['accidents'] = None
            monthly_data['accident_counts'] = None
            monthly_data['neg_edges'] = None
            monthly_data['temporal_node_features'] = None
            monthly_data['temporal_edge_features'] = None
            return monthly_data
        accidents = pd.read_csv(os.path.join(self.data_dir, accident_dir))

        # print(f"accidents : {accidents}")
        # print(f"year : {year}")
        monthly_accidents = accidents[accidents["year"] == year]
        # print(f"monthly_accidents 1: {monthly_accidents}")
        monthly_accidents = monthly_accidents[monthly_accidents["month"] == month]
        # print(f"monthly_accidents 2: {monthly_accidents}")
        monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count", "year", "month"]].values

        # print(f"monthly_accidents 3: {monthly_accidents}")
        pos_edges = torch.Tensor(monthly_accidents[:, :2])
        pos_edge_weights = torch.Tensor(monthly_accidents[:, 2])
        # print(f"pos_edges_before: {pos_edges}")
        pos_edges = pos_edges.long()
        pos_edges, pos_edge_weights = coalesce(pos_edges.T, pos_edge_weights)
        # print(f"pos_edges_mid: {pos_edges}")
        pos_edges = pos_edges.type(torch.int64).T
        # print(f"pos_edges_end: {pos_edges}")
        # sample negative edges from rest of edges in the road network
        all_edges = self.data.edge_index.cpu().T.numpy()
        neg_mask = np.logical_not(np.isin(all_edges, pos_edges.numpy()).all(axis=1))
        neg_edges = all_edges[neg_mask]
        rng = np.random.default_rng(year * 12 + month)
        num_negative_edges = min(max(self.num_negative_edges, pos_edges.shape[0]), neg_edges.shape[0])
        neg_edges = neg_edges[rng.choice(neg_edges.shape[0], num_negative_edges, replace=False)]
        neg_edges = torch.Tensor(neg_edges).type(torch.int64)

        # load the node features of the month
        node_feature_dir = f"{self.state_name}/Nodes/node_features_{year}_{month}.csv"
        if not os.path.exists(os.path.join(self.data_dir, node_feature_dir)):
            new_data = self.data.clone()
            monthly_data['data'] = new_data
            monthly_data['x'] = new_data.x
            monthly_data['edge_index'] = new_data.edge_index
            monthly_data['edge_attr'] = new_data.edge_attr
            monthly_data['accidents'] = None
            monthly_data['accident_counts'] = None
            monthly_data['neg_edges'] = None
            monthly_data['temporal_node_features'] = None
            monthly_data['temporal_edge_features'] = None
            return monthly_data
        node_features = pd.read_csv(os.path.join(self.data_dir, node_feature_dir))
        node_features = node_features[["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]]
        node_features = node_features.fillna(node_features.mean(axis=0))
        node_features = node_features.fillna(0)
        node_features = torch.Tensor(node_features.values)

        # load the edge features 
        edge_feature_dir = os.path.join(self.data_dir, f"{self.state_name}/Edges/edge_features_traffic_{year}.pt")
        if os.path.exists(edge_feature_dir):
            edge_feature_dict = torch.load(edge_feature_dir)

            # for key in ['AADT']: edge_features = torch.stack(edge_features, dim=1)
            column_values = edge_feature_dict['AADT'].coalesce().values()
            column_values_mean = column_values[~torch.isnan(column_values)].mean()
            column_values[torch.isnan(column_values)] = 0 if torch.isnan(column_values_mean) else column_values_mean
            edge_features = column_values.view(-1, 1)
        else:
            edge_features = torch.zeros(self.data.edge_index.shape[1], 1)

        new_data = self.data.clone()

        # normalize node and edge features
        if self.node_feature_mean is not None:
            normalized_node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        else:
            normalized_node_features = node_features

        if self.edge_feature_mean is not None:
            normalized_edge_features = (edge_features - self.edge_feature_mean) / self.edge_feature_std
        else:
            normalized_edge_features = edge_features

        if self.use_dynamic_node_features:
            if new_data.x is None:
                new_data.x = normalized_node_features
            else:
                new_data.x = torch.cat([new_data.x, normalized_node_features], dim=1)

        if self.use_dynamic_edge_features:
            if new_data.edge_attr is None:
                new_data.edge_attr = normalized_edge_features
            else:
                new_data.edge_attr = torch.cat([new_data.edge_attr, normalized_edge_features], dim=1)
        
        monthly_data['data'] = new_data
        monthly_data['x'] = new_data.x
        monthly_data['edge_index'] = new_data.edge_index
        monthly_data['edge_attr'] = new_data.edge_attr
        monthly_data['accidents'] = pos_edges
        monthly_data['accident_counts'] = pos_edge_weights
        monthly_data['neg_edges'] = neg_edges
        monthly_data['temporal_node_features'] = node_features
        monthly_data['temporal_edge_features'] = edge_features

        return monthly_data

    def load_yearly_data(self, year=2022):
        '''
        Return: 
            Valid traffic volume data for a year
            Average the node features over 12 months in the year
        '''
        yearly_data = {}

        # load the edge features 
        edge_feature_name = f"{self.state_name}/Edges/edge_features_traffic_{year}.pt"
        if not os.path.exists(os.path.join(self.data_dir, edge_feature_name)):
            new_data = self.data.clone()
            yearly_data['data'] = new_data
            yearly_data['x'] = new_data.x
            yearly_data['edge_index'] = new_data.edge_index
            yearly_data['edge_attr'] = new_data.edge_attr
            yearly_data['traffic_volume_edges'] = None
            yearly_data['traffic_volume_weights'] = None
            yearly_data['temporal_node_features'] = None
            return yearly_data
        
        edge_feature_dir = os.path.join(self.data_dir, edge_feature_name)
        if not os.path.exists(edge_feature_dir):
            raise ValueError("Edge features not found!")
        
        edge_feature_dict = torch.load(edge_feature_dir)
        edge_indices =  edge_feature_dict['AADT'].coalesce().indices()
        edge_weights = edge_feature_dict['AADT'].coalesce().values()
        mask = ~torch.isnan(edge_weights)
        edge_indices = edge_indices[:, mask]
        edge_indices = edge_indices.type(torch.int64).T
        edge_weights = edge_weights[mask]/1000

        # load the node features of the month
        node_features = []
        for month in range(1, 13):
            month_node_features = pd.read_csv(os.path.join(self.data_dir, f"{self.state_name}/Nodes/node_features_{year}_{month}.csv"))
            month_node_features = month_node_features[["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]]
            month_node_features = month_node_features.fillna(month_node_features.mean(axis=0))
            month_node_features = month_node_features.fillna(0)
            month_node_features = torch.Tensor(month_node_features.values)
            node_features.append(month_node_features)
        node_features = torch.stack(node_features, dim=1)
        node_features = node_features.mean(dim=1)


        new_data = self.data.clone()

        # normalize node and edge features
        if self.node_feature_mean is not None:
            nomalized_node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        
        if self.use_dynamic_node_features:
            if new_data.x is None:
                new_data.x = nomalized_node_features
            else:
                new_data.x = torch.cat([new_data.x, nomalized_node_features], dim=1)
        
        # no loading temporal edge features
        yearly_data['data'] = new_data
        yearly_data['x'] = new_data.x
        yearly_data['edge_index'] = new_data.edge_index
        yearly_data['edge_attr'] = new_data.edge_attr
        yearly_data['traffic_volume_edges'] = edge_indices
        yearly_data['traffic_volume_weights'] = edge_weights
        yearly_data['temporal_node_features'] = node_features

        return yearly_data

    def load_static_edge_features(self):
        edge_feature_dict = torch.load(os.path.join(self.data_dir, f"{self.state_name}/Edges/edge_features.pt"))

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
        if self.state_name == "NV":
            edge_features = torch.concat([edge_features, torch.zeros(2, edge_features.shape[1])], dim=0)
        return edge_features

    def load_static_network(self):
        # Load adjacency matrix
        adj = torch.load(os.path.join(self.data_dir, f"{self.state_name}/adj_matrix.pt"))
        edge_index = adj.coalesce().indices().long()

        data = Data(edge_index=edge_index)

        # load node embeddings
        embedding_dir = os.path.join("./embeddings/", f"{self.node_feature_type}/{self.node_feature_name}")
        if os.path.exists(embedding_dir):
            node_embeddings = np.load(embedding_dir)
            data.x = torch.Tensor(node_embeddings)
            print("Node embeddings loaded!")

        return data

    def compute_feature_mean_std(self, train_years):
        all_node_features = []
        all_edge_features = []
        for year in train_years:
            for month in range(1, 13):
                monthly_data = self.load_monthly_data(year, month)
                # print("monthly_data: ",monthly_data)
                node_features, edge_features = monthly_data['temporal_node_features'], monthly_data['temporal_edge_features']
                all_node_features.append(node_features)
                all_edge_features.append(edge_features)
        
        print("len(all_node_features): ",len(all_node_features))
        print("len(all_edge_features): ",len(all_edge_features))

        all_node_features = torch.cat(all_node_features, dim=0)
        all_edge_features = torch.cat(all_edge_features, dim=0)

        node_feature_mean, node_feature_std = all_node_features.mean(dim=0), all_node_features.std(dim=0)
        edge_feature_mean, edge_feature_std = all_edge_features.mean(dim=0), all_edge_features.std(dim=0)
        print("Successfully computed the mean and std of the features!")
        node_feature_std[node_feature_std == 0] = 1
        edge_feature_std[edge_feature_std == 0] = 1
        return node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std
    
    def get_feature_stats(self):
        return self.node_feature_mean, self.node_feature_std, self.edge_feature_mean, self.edge_feature_std


def load_monthly_data(data, data_dir = "./data", state_name = "MA", num_negative_edges = 100000, year=2022, month=1):
    '''
    Return:
        - node features of the month
        - edge features of the month
        - accidents of the month: every accident labeled with (year_month, node_1_idx, node_2_idx, acc_count)
        - negative edges 
    '''
    # Load accidents
    accident_dir = f"{state_name}/accidents_monthly.csv"
    if not os.path.exists(os.path.join(data_dir, accident_dir)):
        return None, None, None, None, None
    accidents = pd.read_csv(os.path.join(data_dir, accident_dir))

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
    num_negative_edges = min(max(num_negative_edges, pos_edges.shape[0]), neg_edges.shape[0])
    neg_edges = neg_edges[rng.choice(neg_edges.shape[0], num_negative_edges, replace=False)]
    neg_edges = torch.Tensor(neg_edges).type(torch.int64)

    # load the node features of the month
    node_feature_dir = f"{state_name}/Nodes/node_features_{year}_{month}.csv"
    if not os.path.exists(os.path.join(data_dir, node_feature_dir)):
        return None, None, None, None, None
    node_features = pd.read_csv(os.path.join(data_dir, node_feature_dir))
    node_features = node_features[["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]]
    node_features = node_features.fillna(node_features.mean(axis=0))
    node_features = node_features.fillna(0)
    node_features = torch.Tensor(node_features.values)

    # load the edge features 
    edge_feature_dir = os.path.join(data_dir, f"{state_name}/Edges/edge_features_traffic_{year}.pt")
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

def load_yearly_data(data_dir = "./data", state_name = "MA", year=2022):
    '''
    Return: 
        Valid traffic volume data for a year
        Average the node features over 12 months in the year
    '''
    # load the edge features 
    edge_feature_name = f"{state_name}/Edges/edge_features_traffic_{year}.pt"
    if not os.path.exists(os.path.join(data_dir, edge_feature_name)):
        return None, None, None
    edge_feature_dir = os.path.join(data_dir, edge_feature_name)
    if not os.path.exists(edge_feature_dir):
        raise ValueError("Edge features not found!")
    
    edge_feature_dict = torch.load(edge_feature_dir)
    edge_indices =  edge_feature_dict['AADT'].coalesce().indices()
    edge_weights = edge_feature_dict['AADT'].coalesce().values()
    mask = ~torch.isnan(edge_weights)
    edge_indices = edge_indices[:, mask]
    edge_indices = edge_indices.type(torch.int64).T
    edge_weights = edge_weights[mask]/1000

    # load the node features of the month
    node_features = []
    for month in range(1, 13):
        month_node_features = pd.read_csv(os.path.join(data_dir, f"{state_name}/Nodes/node_features_{year}_{month}.csv"))
        month_node_features = month_node_features[["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]]
        month_node_features = month_node_features.fillna(month_node_features.mean(axis=0))
        month_node_features = month_node_features.fillna(0)
        month_node_features = torch.Tensor(month_node_features.values)
        node_features.append(month_node_features)
    node_features = torch.stack(node_features, dim=1)
    node_features = node_features.mean(dim=1)

    return edge_indices, edge_weights, node_features

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
    if state_name == "NV":
        edge_features = torch.concat([edge_features, torch.zeros(2, edge_features.shape[1])], dim=0)
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
