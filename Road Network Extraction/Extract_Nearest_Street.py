
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#--------------- Initializing Paramaters ----------#

path = os.getcwd()

path_nodes = path + "/Stats/Nodes/"
path_edges = path + "/Stats/Edges/"

#---------------- Load Files ----------------#

nodes_df = pd.read_csv(path + "/Stats/Stats_Nodes_MA.csv")
edges_df = pd.read_csv(path + "/Stats/Stats_Edges_MA.csv")

with open(path + "/mass_fatality_lat_lon.txt",'r',encoding='utf-8') as f:
    lines = f.readlines()

#---------------- Clean Files ---------------#

edges_df = edges_df[["node_1","node_2"]].drop_duplicates()

nodes_df = nodes_df[["node_id","x","y"]].drop_duplicates()


df = {"lat":[],"lon":[]}
for line in lines[1:]:
    df["lat"].append(line.strip().split(",")[-2])
    df["lon"].append(line.strip().split(",")[-1])
df = pd.DataFrame(df)

df = df[(df["lat"] != "") & (df["lon"] != "")].reset_index(drop=True)

df["lat"] = pd.to_numeric(df["lat"])
df["lon"] = pd.to_numeric(df["lon"])


#----------------- Functions -------------------#

def extract_nearest_street(edges_df,lat,lon):
    '''
    Extract the nodes of the nearest street given a latlong coordinate
    Methodology:
        Calculate the distance between 2 nodes
        Calculate the sum of distances between the point 
        and two nodes
        Extract the nodes/street with the minimum distance
    Parameters:
        edges_df (dataframe): details of the edges
        lat (float): latitude of the point
        lon (float): longitude of the point
    Returns:
        node 1, node 2
    '''

    edges_df["street_dist_node_1"] = np.sqrt((lon - edges_df["node_1_x"])**2 + (lat - edges_df["node_1_y"])**2)
    edges_df["street_dist_node_2"] = np.sqrt((lon - edges_df["node_2_x"])**2 + (lat - edges_df["node_2_y"])**2)

    edges_df["street_dist_node_1_plus_node_2"] = edges_df["street_dist_node_1"] + edges_df["street_dist_node_2"]

    edges_df["street_dist_diff"] = np.abs(edges_df["street_dist_node_1_plus_node_2"] - edges_df["street_dist"])

    min_df = edges_df[edges_df["street_dist_diff"] == edges_df["street_dist_diff"].min()].reset_index(drop=True)

    return min_df.loc[0,"node_1"],min_df.loc[0,"node_2"]


#------------------ Extract the nearest street details ------------------#

# Merge the latlong coordinates of the nodes of all the edges
edges_df = pd.merge(edges_df,nodes_df,left_on="node_1",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
edges_df = edges_df.rename(columns={"x":"node_1_x","y":"node_1_y"})

edges_df = pd.merge(edges_df,nodes_df,left_on="node_2",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
edges_df = edges_df.rename(columns={"x":"node_2_x","y":"node_2_y"})

# Calculate the distance between 2 nodes
edges_df["street_dist"] = np.sqrt((edges_df["node_2_x"] - edges_df["node_1_x"])**2 + (edges_df["node_2_y"] - edges_df["node_1_y"])**2)


# Extract the nodes of the nearest street from the given point
print("Extract Nearest Street:")
df["node_1"] = 0
df["node_2"] = 0
for i in tqdm(range(df.shape[0])):
    df.loc[i,"node_1"] = extract_nearest_street(edges_df,df.loc[i,"lat"],df.loc[i,"lon"])[0]
    df.loc[i,"node_2"] = extract_nearest_street(edges_df,df.loc[i,"lat"],df.loc[i,"lon"])[1]

# Save the file
df.to_csv(path + "/Nearest_Streets.csv",index=False)


















