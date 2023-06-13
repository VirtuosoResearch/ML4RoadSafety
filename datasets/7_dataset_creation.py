
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os

import torch
from os.path import dirname

from torch.sparse import FloatTensor
from tqdm import tqdm

#--------------- Initializing Paramaters ----------#

path = dirname(os.getcwd())

state_name = "MA"


#---------------- Functions -------------------#


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

#--------------- Nodes --------------------------#

df_nodes = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv", low_memory=False)

df_nodes.columns = ["node_id","lat","lon"]

#--------------- Edges -------------------------#

df_edges = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Edges_" + state_name + ".csv", low_memory=False)

df_edges = df_edges.drop(["name"],axis=1)

print("\nOne Hot Encode Categorical Features")

print(np.unique(df_edges["oneway"]))

# Oneway
df_edges["oneway"] = df_edges["oneway"].apply(lambda x: int(x))

# Highway
highway_types = []
for highway_type in np.unique(df_edges["highway"]):
    if(highway_type[0] == '['):
        dummy = "".join([char for char in highway_type if char not in ["[","]","'"]])
        highway_types += dummy.split(", ")
    else:
        highway_types += [highway_type]

highway_types = list(np.unique(highway_types))
print(len(highway_types))

for highway_type in highway_types:
    df_edges[highway_type] = df_edges["highway"].apply(lambda x: 1 if highway_type in x else 0)

df_edges = df_edges.drop(columns=["highway"])

#------------------- Accidents ----------------------#
print("\nAccident Records")
df_accidents = pd.read_csv(path + "/Accidents/" + state_name + "/Accidents_Nearest_Street_" + state_name + ".csv", low_memory=False)

df_accidents["accident_date"] = pd.to_datetime(df_accidents["accident_date"])#,format='%Y-%m-%d')

df_accidents["year"] = df_accidents["accident_date"].dt.year
df_accidents["month"] = df_accidents["accident_date"].dt.month
df_accidents["day"] = df_accidents["accident_date"].dt.day

df_accidents = df_accidents.sort_values(["year","month"],ascending=[True,True])


df_accidents = df_accidents.groupby(["year","month","node_1","node_2"],as_index=False)["acc_count"].sum()

df_accidents["node_1_idx"] = ""
df_accidents["node_2_idx"] = ""
for i in tqdm(range(df_accidents.shape[0])):
    df_accidents.loc[i,"node_1_idx"] = df_nodes[df_nodes['node_id'] == df_accidents.loc[i,"node_1"]].index[0]
    df_accidents.loc[i,"node_2_idx"] = df_nodes[df_nodes['node_id'] == df_accidents.loc[i,"node_2"]].index[0]


df_accidents.to_csv(path + "/Accidents/" + state_name + "/Accidents_Nearest_Street_" + state_name + "_Monthly.csv",index=False)
df_accidents.to_csv(path + "/Final_Graphs/" + state_name + "/Accidents_Nearest_Street_" + state_name + "_Monthly.csv",index=False)


#-------------- Set the dates --------------------#

start_date = '2015-01-01'
end_date = '2023-04-01'

date_range = pd.date_range(start=start_date, end=end_date, freq='D')
date_range = pd.date_range(start=start_date, end=end_date, freq='M')

dates_df = pd.DataFrame({'year': date_range.year,
                   'month': date_range.month})
                    # 'day': date_range.day})


#------------------- Traffic ----------------------#

df_traffic = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT.csv")

#------------------- Weather ----------------------#

df_weather = pd.read_csv(path + "/Weather_Features/" + state_name + "/" + state_name + "_Weather_Features.csv")

df_weather["time"] = pd.to_datetime(df_weather["time"])

df_weather["year"] = df_weather["time"].dt.year
df_weather["month"] = df_weather["time"].dt.month

df_weather = df_weather.sort_values(["year","month"],ascending=[True,True])


#------------------- Adjacency Matrix ----------------------#


print("\nAdjacency Matrix")

# N*N*F
adj_matrix = create_adjacency_matrix(df_nodes, df_edges)
torch.save(adj_matrix, path + "/Final_Graphs/" + state_name + '/adj_matrix.pt')


print("\nCreate Node Features")
for i in tqdm(range(len(dates_df))):

    year = dates_df.loc[i,"year"]
    month = dates_df.loc[i,"month"]

    # print(f"\n******* Date - {year} - {month} ************")
    
    weather_filtered_df = df_weather[(df_weather["year"] == year) & (df_weather["month"] == month)]
    weather_filtered_df = weather_filtered_df[["node_id","tavg","tmin","tmax","prcp","wspd","pres"]]
    
    df_nodes_time = pd.merge(df_nodes, weather_filtered_df, on=["node_id"],how="left").drop_duplicates()

    df_nodes_time.to_csv(path + "/Final_Graphs/" + state_name + "/Nodes/node_features_" + str(year) + "_" + str(month) + ".csv",index=False)




print("\nCreate Edge Features")


edge_features_time = create_edge_features(df_nodes, df_edges)
torch.save(edge_features_time, path + "/Final_Graphs/" + state_name + '/Edges/edge_features.pt')


for year in np.unique(dates_df["year"]):

    month = 1

    print(f"\n******* Date - {year} ************")

    traffic_filtered_df = df_traffic[(df_traffic["year"] == year)].drop(columns=["year"])


    df_edges_time = pd.merge(df_edges, traffic_filtered_df, on=["node_1","node_2"],how="left").drop_duplicates()

    # N*N*F
    edge_features_time = create_edge_features(df_nodes, df_edges_time)
    torch.save(edge_features_time, path + "/Final_Graphs/" + state_name + '/Edges/edge_features_traffic_' + str(year) + '.pt')




