
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#--------------- Initializing Paramaters ----------#

path = os.getcwd()

state_name = "NYC"

path_nodes = path + "/Stats/" + state_name + "/Nodes/"
path_edges = path + "/Stats/" + state_name + "/Edges/"

#---------------- Load Files ----------------#

nodes_df = pd.read_csv(path + "/Stats/" + state_name + "/Stats_Nodes_" + state_name + ".csv")
edges_df = pd.read_csv(path + "/Stats/" + state_name + "/Stats_Edges_" + state_name + ".csv")

# Accident Data
df = pd.read_csv(path + "/nyc_crash.csv")

#---------------- Clean Files ---------------#

edges_df = edges_df[["node_1","node_2"]].drop_duplicates()

nodes_df = nodes_df[["node_id","x","y"]].drop_duplicates()

df = df[(df["lat"].isnull() == False) & (df["lon"].isnull() == False)].reset_index(drop=True)

df["lat"] = pd.to_numeric(df["lat"])
df["lon"] = pd.to_numeric(df["lon"])

df = df[(df["lat"] != 0) & (df["lon"] != 0)].reset_index(drop=True)


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
for j in range(int(df.shape[0]/10000)):
    print(f"**** {j} ****")
    for i in tqdm(range(j*10000, (j+1)*10000)):
        df.loc[i,"node_1"],df.loc[i,"node_2"] = extract_nearest_street(edges_df,df.loc[i,"lat"],df.loc[i,"lon"])

    # Save the file
    dummy = df.iloc[j*10000 : (j+1)*10000, :]
    dummy.to_csv(path + "/Nearest_Streets_" + state_name + "_" + str(j) + ".csv",index=False)


for i in tqdm(range((j+1)*10000,df.shape[0])):
    df.loc[i,"node_1"],df.loc[i,"node_2"] = extract_nearest_street(edges_df,df.loc[i,"lat"],df.loc[i,"lon"])

# Save the file
dummy = df.iloc[(j+1)*10000 :, :]
dummy.to_csv(path + "/Nearest_Streets_" + state_name + "_" + str(j+1) + ".csv",index=False)


















