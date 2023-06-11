#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from os.path import dirname

from Road_Network_Functions import *

#--------------- Initializing Paramaters ----------#

path = dirname(os.getcwd())

state_name = "MA"

path_stats = path + "/Road_Networks/" + state_name + "/"


#------------- Load Level-Wise Road Networks ---------------#

print("\nRoad Network of every Level:")

print("\tCities")
node_df, edge_df = concat_files_one_subfolder(path_stats + "Harvard Dataverse/" + state_name + "-cities-street_networks-node_edge_lists/")

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_cities.csv",index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_cities.csv",index=False)

print("\tCounties")
node_df, edge_df = concat_files_one_subfolder(path_stats + "Harvard Dataverse/" + state_name + "-counties-street_networks-node_edge_lists/")

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_counties.csv",index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_counties.csv",index=False)

print("\tNeighborhoods")
node_df, edge_df = concat_files_two_subfolder(path_stats + "Harvard Dataverse/" + state_name + "-neighborhoods-street_networks-node_edge_lists/")

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_neighborhoods.csv",index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_neighborhoods.csv",index=False)


print("\tTracts")
node_df, edge_df = concat_files_one_subfolder(path_stats + "Harvard Dataverse/" + state_name + "-tracts-street_networks-node_edge_lists/")

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_tracts.csv",index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_tracts.csv",index=False)


print("\tUrbanized Areas")
node_df, edge_df = concat_files_one_subfolder(path_stats + "Harvard Dataverse/" + state_name + "-urbanized_areas-street_networks-node_edge_lists/")

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_urbanized_areas.csv",index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_urbanized_areas.csv",index=False)


#----------- Concatenate the data --------------#

print("\nAppend all files")

print("\tNodes")
concat_files(path_stats + "Road_Network_Level/Nodes/", path_stats + "Road_Network_Nodes_" + state_name + ".csv")

print("\tEdges")
concat_files(path_stats + "Road_Network_Level/Edges/", path_stats + "Road_Network_Edges_" + state_name + ".csv")


print("\nRemove Duplicates and Save:")

print("\tNodes")
df_nodes = pd.read_csv(path_stats + "Road_Network_Nodes_" + state_name + ".csv",low_memory=False)

df_nodes = df_nodes.drop_duplicates(["node_id"],keep="last")
df_nodes = df_nodes[["node_id","x","y"]]

df_nodes.to_csv(path_stats + "Road_Network_Nodes_" + state_name + ".csv",index=False)


print("\tEdges")
df_edges = pd.read_csv(path_stats + "Road_Network_Edges_" + state_name + ".csv",low_memory=False)

df_edges = df_edges.drop_duplicates(["node_1","node_2"],keep="last")
df_edges = df_edges[["node_1","node_2","oneway","highway","name","length"]]

df_edges.to_csv(path_stats + "Road_Network_Edges_" + state_name + ".csv",index=False)


#------------------------#






















