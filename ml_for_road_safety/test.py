# %%
from data_loaders import load_network_with_accidents

data, split_edge = load_network_with_accidents(data_dir="./data", state_name="MA", 
                                                   train_years=[2002], train_months=[1, 2, 3, 4],
                                                   valid_years=[2002], valid_months=[5, 6, 7, 8],
                                                   test_years=[2002], test_months=[9, 10, 11, 12],
                                                   num_negative_edges = 1000000,
                                                   feature_type="verse", feature_name = "MA_ppr_128.npy")

pos_train_edge = split_edge['train']['edge'].numpy()
pos_valid_edge = split_edge['valid']['edge'].numpy()
neg_valid_edge = split_edge['valid']['edge_neg'].numpy()
pos_test_edge = split_edge['test']['edge'].numpy()
neg_test_edge = split_edge['test']['edge_neg'].numpy()

all_edges = data.edge_index.T.numpy()


# %%
from data_loaders import load_static_network, load_monthly_data

state_name = "NV"

state_to_train_years = {
"DE": [2009, 2010, 2011, 2012],
"IA": [2013, 2014, 2015, 2016],
"IL": [2012, 2013, 2014],
"MA": [2002, 2003, 2004, 2005, 2006, 2007, 2008],
"MD": [2015, 2016, 2017],
"MN": [2015, 2016, 2017],
"MT": [2016, 2017],
"NV": [2016, 2017],
}

state_to_valid_years = {
"DE": [2013, 2014, 2015, 2016, 2017],
"IA": [2017, 2018, 2019],
"IL": [2015, 2016, 2017],
"MA": [2009, 2010, 2011, 2012, 2013, 2014, 2015],
"MD": [2018, 2019],
"MN": [2018, 2019],
"MT": [2018],
"NV": [2018],
}

state_to_test_years = {
"DE": [2018, 2019, 2020, 2021, 2022],
"IA": [2020, 2021, 2022],
"IL": [2018, 2019, 2020, 2021],
"MA": [2016, 2017, 2018, 2019, 2020, 2021, 2022],
"MD": [2020, 2021, 2022],
"MN": [2020, 2021, 2022],
"MT": [2019, 2020],
"NV": [2019, 2020],
}

data = load_static_network(data_dir="./data", state_name=state_name, feature_type="node2vec", feature_name = f"{state_name}_128.npy")


train_records = 0; training_weights = 0
for year in state_to_train_years[state_name]:
    for month in range(1, 13):
        pos_edges, pos_edge_weights, _, _, _ = load_monthly_data(data, data_dir="./data", state_name=state_name, year=year, month = month)
        train_records += pos_edges.size(0) if pos_edges is not None else 0
        training_weights += pos_edge_weights.sum() if pos_edge_weights is not None else 0
print("Training records: ", train_records)

valid_records = 0; validation_weights = 0
for year in state_to_valid_years[state_name]:
    for month in range(1, 13):
        pos_edges, pos_edge_weights, _, _, _ = load_monthly_data(data, data_dir="./data", state_name=state_name, year=year, month = month)
        valid_records += pos_edges.size(0) if pos_edges is not None else 0
        validation_weights += pos_edge_weights.sum() if pos_edge_weights is not None else 0
print("Validation records: ", valid_records)


test_records = 0; testing_weights = 0
for year in state_to_test_years[state_name]:
    for month in range(1, 13):
        pos_edges, pos_edge_weights, _, _, _ = load_monthly_data(data, data_dir="./data", state_name=state_name, year=year, month = month)
        test_records += pos_edges.size(0) if pos_edges is not None else 0
        testing_weights += pos_edge_weights.sum() if pos_edge_weights is not None else 0
print("Testing records: ", test_records)

print("Avg. count", (training_weights + validation_weights + testing_weights)/(train_records + valid_records + test_records))
print("Positive rate: ", (train_records + valid_records + test_records)/(data.num_edges * (len(state_to_train_years[state_name]) + len(state_to_valid_years[state_name]) + len(state_to_test_years[state_name]))))

# %%
import numpy as np

def organize_edges(edges):
    mask = edges[:, 0] > edges[:, 1]
    edges[mask, 0], edges[mask, 1] = edges[mask, 1], edges[mask, 0]
    return edges

pos_train_edge = organize_edges(pos_train_edge)
pos_train_edge_set = set([
    (pos_train_edge[i, 0], pos_train_edge[i, 1]) for i in range(pos_train_edge.shape[0])
])

pos_valid_edge = organize_edges(pos_valid_edge)
pos_valid_edge_set = set([
    (pos_valid_edge[i, 0], pos_valid_edge[i, 1]) for i in range(pos_valid_edge.shape[0])
])

pos_test_edge = organize_edges(pos_test_edge)
pos_test_edge_set = set([
    (pos_test_edge[i, 0], pos_test_edge[i, 1]) for i in range(pos_test_edge.shape[0])
])

print(len(pos_train_edge_set), len(pos_valid_edge_set), len(pos_test_edge_set))

print(
    len(pos_train_edge_set.intersection(pos_valid_edge_set))
)

print(
    len(pos_train_edge_set.intersection(pos_test_edge_set))
)

# import scipy.sparse as sp
# import numpy as np
# train_adj = sp.coo_matrix(
#     (np.ones(pos_train_edge.shape[0]), (pos_train_edge[:, 0], pos_train_edge[:, 1])),
#     )
# valid_adj = sp.coo_matrix(
#     (np.ones(pos_valid_edge.shape[0]), (pos_valid_edge[:, 0], pos_valid_edge[:, 1])),
#     )


# %%
import os
import torch
import pandas as pd

data_dir = "./data"
state_name = "MA"

adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
# n = 285942 and m = 706402
edge_index = adj.coalesce().indices()
# %%
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data
import torch_geometric.transforms as T

data = Data(edge_index=edge_index)
print(is_undirected(data.edge_index))
data = T.ToUndirected()(data)
num_nodes = data.num_nodes
# Data(edge_index=[2, 746653])

# %%
import numpy as np
import pandas as pd
data_dir = "./data"
state_name = "DE"
accidents = pd.read_csv(os.path.join(data_dir, f"{state_name}/accidents_monthly.csv"))

# %%
training_edges = []
training_edge_weights = []

for year in range(2002, 2013):
    monthly_accidents = accidents[accidents["year"] == year]
    monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count"]].values
    training_edges.append(monthly_accidents[:, :2])
    training_edge_weights.append(monthly_accidents[:, 2])

training_edges = torch.Tensor(np.concatenate(training_edges, axis=0))
training_edge_weights = torch.Tensor(np.concatenate(training_edge_weights, axis=0))

valid_edges = []
valid_edge_weights = []

for year in range(2013, 2018):
    monthly_accidents = accidents[accidents["year"] == year]
    monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count"]].values
    valid_edges.append(monthly_accidents[:, :2])
    valid_edge_weights.append(monthly_accidents[:, 2])

valid_edges = torch.Tensor(np.concatenate(valid_edges, axis=0))
valid_edge_weights = torch.Tensor(np.concatenate(valid_edge_weights, axis=0))

test_edges = []
test_edge_weights = []

for year in range(2018, 2023):
    monthly_accidents = accidents[accidents["year"] == year]
    monthly_accidents = monthly_accidents[["node_1_idx", "node_2_idx", "acc_count"]].values
    test_edges.append(monthly_accidents[:, :2])
    test_edge_weights.append(monthly_accidents[:, 2])

test_edges = torch.Tensor(np.concatenate(test_edges, axis=0))
test_edge_weights = torch.Tensor(np.concatenate(test_edge_weights, axis=0))


'''
Return:
    - data: adj and full_adj
        use full network as adj
        use only training positive edges as adj
    - splits: training edges, validation edges, validation negative edges, test edges, test negative edges
'''
# %%
from torch_geometric.utils import coalesce

training_edges, training_edge_weights = coalesce(training_edges.T, training_edge_weights)
training_edges = training_edges.type(torch.int64)

valid_edges, valid_edge_weights = coalesce(valid_edges.T, valid_edge_weights)
valid_edges = valid_edges.type(torch.int64)

test_edges, test_edge_weights = coalesce(test_edges.T, test_edge_weights)
test_edges = test_edges.type(torch.int64)

# %%
from ogb.linkproppred import PygLinkPropPredDataset

dataset = PygLinkPropPredDataset(name='ogbl-collab')
split_edge = dataset.get_edge_split()

# %%
# split_edge['train']['edge'].shape
valid_edges = split_edge['valid']['edge']
neg_valid_edge = split_edge['valid']['edge_neg']

test_edges = split_edge['test']['edge']
neg_test_edge = split_edge['test']['edge_neg']



# %%
import os
import torch
import pandas as pd

data_dir = "./data"
state_name = "MA"
node_features = pd.read_csv(os.path.join(data_dir, f"{state_name}/Nodes/node_features_2023_3.csv"))

pd_isna = node_features.isna()
print(pd_isna["tavg"].mean())
print(pd_isna["tmin"].mean())
print(pd_isna["tmax"].mean())
print(pd_isna["prcp"].mean())
print(pd_isna["wspd"].mean())
print(pd_isna["pres"].mean())

# %%
import os
import torch
import pandas as pd

data_dir = "./data"
state_name = "MD"
edge_feature_dict = torch.load(os.path.join(data_dir, f"{state_name}/Edges/edge_features_traffic_2012.pt"))

# edge_features = []
# for key in ['oneway', 'access_ramp', 'bus_stop', 'crossing', 'disused', 'elevator', 'escape', 'living_street', 'motorway', 'motorway_link', 'primary', 'primary_link', 'residential', 'rest_area', 'road', 'secondary', 'secondary_link', 'stairs', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified', 'unsurfaced']:
#     column_values = edge_feature_dict[key].coalesce().values()
#     # print(torch.isnan(column_values).sum())
#     column_values[torch.isnan(column_values)] = 0
#     edge_features.append(column_values)

# edge_features = torch.stack(edge_features, dim=1)

# edge_lengths = edge_feature_dict['length'].coalesce().values()
# length_mean = edge_lengths[~torch.isnan(edge_lengths)].mean()
# edge_lengths[torch.isnan(edge_lengths)] = length_mean



# %%
import os
import torch
import pandas as pd
from torch_geometric.data import Data
from data_loaders import load_static_network, load_static_edge_features


data_dir = "./data"
state_name = "NV"

data = load_static_network(data_dir="./data", state_name=state_name, 
                               feature_type="verse", 
                               feature_name = "MA_ppr_128.npy")
data.edge_attr = load_static_edge_features(data_dir="./data", state_name=state_name)
# %%
from torch_geometric.loader import NeighborLoader

train_loader = NeighborLoader(data,
                               num_neighbors=[-1, -1, -1], batch_size=4098,
                               num_workers=12)

for batch in train_loader:
    print(batch)
    break

# %%
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from data_loaders import load_monthly_data

pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(data, data_dir=data_dir, state_name=state_name, year=2002, month = 1, num_negative_edges=10000)

# %%
data.pos_edge_index = pos_edges.T
data.pos_edge_weight = pos_edge_weights
data.neg_edge_index = neg_edges.T

loader = GraphSAINTRandomWalkSampler(data, batch_size=16*1024,
                                        walk_length=3,
                                        num_steps=20,
                                        sample_coverage=0,
                                    ) # from torch_geometric.data import GraphSAINTRandomWalkSampler


# %%
from data_loaders import load_static_edge_features

edge_features = load_static_edge_features(data_dir, state_name)


# %%
from data_loaders import load_yearly_data

edges, edge_weights, node_features = load_yearly_data(data_dir, state_name, year = 2002)
print(edges.shape, edge_weights.shape, node_features.shape)


# %%
import numpy as np

node_features = np.load("./embeddings/centrality/NV_6.npy")

print(node_features[:, 0].mean())
print(node_features[:, 1].mean())
print(node_features[:, 2].mean())
# %%
import requests

doi = "10.7910/DVN/V71K5R"
file_url = "https://drive.google.com/uc?export=download&id=1CWIFTAYbpPgheqyiGro1ZAnLRwrrwTOz"

response = requests.get(file_url)
# data = response.json()

# download_url = data["data"]["latestVersion"]["files"][0]["dataFile"]["downloadURL"]
