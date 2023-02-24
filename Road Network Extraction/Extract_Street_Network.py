
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import osmnx as ox
import os
from os.path import dirname

#--------------- Initializing Paramaters ----------#

path = os.getcwd()

path_nodes = path + "/Stats/Nodes/"
path_edges = path + "/Stats/Edges/"

node_columns = ['y', 'x', 'street_count']
edge_columns = ['osmid', 'lanes', 'name', 'highway', 'width', 'oneway', 'reversed', 'length', 'geometry']

#------------------ Methodology -----------------#

'''
Dividing MA into big rectangles
Manually extracting the coordinates from google maps

Left Rectangle:
Top left: 42.742815, -73.271423
Top right: 42.706018, -71.290596
Bottom right: 42.019642, -71.377115
Bottom left: 42.050081, -73.494551


Right Rectangle:
# Top left: 42.706018, -71.290596
# Top right: 42.891180, -70.826016
# Bottom right: 41.839008, -69.947745
# Bottom Left: 41.468262, -71.381759

Create a rectangle from the above coordinates

Use API from OpenStreetMaps (osmnx) to extract street networks

Divide the above rectangles into multiple smaller rectangles
to avoid computation issues

'''


#-------------------- Functions -----------------------#

def extract_stats(graph):
    '''
    Extract the node and edge features from OpenStreetMaps
    Parameters:
        graph: street network graph of the region
    Returns:
        node_df (dataframe): stores features of the nodes
        edge_df (dataframe): stores features of the edges
    '''

    # Extract Features of Nodes
    node_df = pd.DataFrame(graph.nodes(data=True))
    node_df.columns = ["node_id","dict"]
    for col in node_df["dict"][0].keys():
        node_df[col] = node_df["dict"].apply(lambda x: x[col])
    node_df = node_df.drop(["dict"],axis=1)

    # Extract Features of Edges
    edge_df = pd.DataFrame(graph.edges(keys=True, data=True))
    edge_df.columns = ["node_1","node_2","random","dict"]
    for col in edge_df["dict"][0].keys():
        edge_df[col] = edge_df["dict"].apply(lambda x: x[col] if col in x.keys() else np.NaN)
    edge_df = edge_df.drop(["dict"],axis=1)

    return node_df, edge_df


def save_stats(lat_coords, long_coords, node_columns, edge_columns, 
               count, actual_count, path_nodes, path_edges):
    '''
    Extract the Graph from OpenStreetMaps, 
    extracts stats and saves them
    Parameters:
        lat_coords (list): range of latitudes in the rectangle
        lon_coords (list): range of longitudes in the rectangle
        node_columns (list): features of nodes
        edge_columns (list): features of edges
        actual_count (int): counter of the region being traversed
        count (int): counter of the region being traversed 
                            and has data which will be saved
        path_nodes (str): directory of the path to save node features
        path_edges (str): directory of the path to save edge features
    Returns:
        actual_count (int): final count of the regions traversed
        count (int): final count of the regions being saved
    '''

    for i in range(len(lat_coords)-1):
        for j in range(len(long_coords)-1):
            if(actual_count%10 == 0):
                print(f"\tActual Count: {actual_count}, Count: {count}")
            try:
                # Extract Graph
                graph = ox.graph_from_bbox(lat_coords[i], lat_coords[i+1], 
                                           long_coords[j], long_coords[j+1], 
                                           network_type='drive')
                
                # Extract Stats
                node_df, edge_df = extract_stats(graph)
                
                # Save
                node_df.to_csv(path_nodes + "Stats_Nodes_" + str(count) + ".csv",index=False)
                edge_df.to_csv(path_edges + "Stats_Edges_" + str(count) + ".csv",index=False)

                # Update Counts
                count += 1
                actual_count += 1
            except:
                actual_count += 1

    return count, actual_count


def concat_files(path, final_file_name):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''
    count = 0
    for file_name in os.listdir(path):
        try:
            df = pd.concat([df,pd.read_csv(path + file_name)])
        except:
            df = pd.read_csv(path + file_name)

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(final_file_name,index=False)


#------------------ Extracting and Saving features ------------------#

count = 0
actual_count = 0

# Left rectangle
print("Left Rectangle:")
lat_coords_left = np.arange(42.050081,42.706018 + 0.1,0.1)
long_coords_left = np.arange(-73.271423,-71.290596 + 0.1,0.1)

count, actual_count = save_stats(lat_coords_left, long_coords_left, 
                                 node_columns, edge_columns, 
                                 count, actual_count, 
                                 path_nodes, path_edges)

# Right rectangle
print("Right Rectangle:")
lat_coords_right = np.arange(41.468262,42.891180 + 0.1,0.1)
long_coords_right = np.arange(-71.381759,-69.947745 + 0.1,0.1)

count, actual_count = save_stats(lat_coords_right, long_coords_right, node_columns, edge_columns, 
                                    count, actual_count, path_nodes, path_edges)


# Concat all the files
print("Nodes:")
concat_files(path_nodes, path + "/Stats/Stats_Nodes_MA.csv")

print("Edges:")
concat_files(path_edges, path + "/Stats/Stats_Edges_MA.csv")












