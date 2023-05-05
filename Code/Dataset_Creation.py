
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os

import torch


from Dataset_Functions import *

#--------------- Initializing Paramaters ----------#

path = os.getcwd()

state_name = "MD"

#--------------- Nodes --------------------------#

df_nodes = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv")

df_nodes.columns = ["node_id","lat","lon"]

#--------------- Edges -------------------------#

df_edges = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Edges_" + state_name + ".csv")

df_edges = df_edges.drop(["name"],axis=1)

df_edges["oneway"] = df_edges["oneway"].apply(lambda x: int(x))

#------------------- Accidents ----------------------#

df_accidents = pd.read_csv(path + "/Accidents/" + state_name + "/Accidents_Nearest_Street_" + state_name + ".csv")
df_accidents["accident_count"] = 1
df_accidents = df_accidents.groupby(["year","month","node_1","node_2"],as_index=False)["accident_count"].sum()

# df_accidents.sort_values(["accident_count"],ascending=False)




print("\nOne Hot Encode Categorical Features")

categorical_feats = ["highway"]
df_edges = one_hot_encode_features(df_edges, categorical_feats)



#-------------- Set the dates --------------------#

start_date = '2015-01-01'
end_date = '2023-04-01'

date_range = pd.date_range(start=start_date, end=end_date, freq='D')
date_range = pd.date_range(start=start_date, end=end_date, freq='M')

dates_df = pd.DataFrame({'year': date_range.year,
                   'month': date_range.month})
                    # 'day': date_range.day})



print("\nAdjacency Matrix")
adj_matrix = create_adjacency_matrix(df_nodes, df_edges)
torch.save(adj_matrix, path + "/Dataset/" + state_name + '/adj_matrix.pt')


print("\nCreate Node Features")
node_features = create_node_features(df_nodes)
torch.save(node_features, path + "/Dataset/" + state_name + '/node_features.pt')



print("\nCreate Edge Features")

for i in range(len(dates_df)):

    year = dates_df.loc[i,"year"]
    month = dates_df.loc[i,"month"]

    print(f"\n******* Date - {year} - {month} ************")

    accident_filtered_df = df_accidents[(df_accidents["year"] == year) & (df_accidents["month"] == month)]
    accident_filtered_df = accident_filtered_df[["node_1","node_2","accident_count"]].reset_index(drop=True)

    print(f"Edges having accidents: {accident_filtered_df.shape[0]}")


    df_edges_time = pd.merge(df_edges, accident_filtered_df, on=["node_1","node_2"],how="left").fillna(0).drop_duplicates()


    edge_features_time = create_edge_features(df_nodes, df_edges_time)
    torch.save(edge_features_time, path + "/Dataset/" + state_name + '/edge_features_' + str(year) + '_' + str(month) + '.pt')




# Save node_features, edge_features, and adj_matrix tensors
# node_features = torch.load(path + "/Dataset/" + state_name + '/node_features.pt')
# edge_features = torch.load(path + "/Dataset/" + state_name + '/edge_features_2013_1.pt')
# adj_matrix = torch.load(path + "/Dataset/" + state_name + '/adj_matrix.pt')



