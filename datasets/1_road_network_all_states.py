import pandas as pd
import numpy as np
import os
from os.path import dirname

# Combine all files in a directory and save them into a final file
def concat_files(path, final_file_name):
    '''
    Combines all files in a directory and saves them into a single file.
    
    Parameters:
        path (str): Directory containing independent files.
        final_file_name (str): Path for the final file to be saved.
    
    Returns:
        None
    '''
    count = 0
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            try:
                # Concatenate the current file with the existing DataFrame
                df = pd.concat([df, pd.read_csv(os.path.join(path, file_name), low_memory=False)])
            except:
                # If no DataFrame exists yet, create it
                df = pd.read_csv(os.path.join(path, file_name), low_memory=False)
            count += 1
    # Remove duplicates and reset index
    df = df.drop_duplicates().reset_index(drop=True)
    # Save the combined DataFrame to the final file
    df.to_csv(final_file_name, index=False)

# Combine node and edge data from subfolders
def concat_files_subfolder(path, node_file_name="node_list.csv", edge_file_name="edge_list.csv"):
    '''
    Recursively searches subdirectories for node and edge files, combines them and returns the dataframes.
    
    Parameters:
        path (str): Directory to search for node and edge files.
        node_file_name (str): Name of the node file to search for. Default is "node_list.csv".
        edge_file_name (str): Name of the edge file to search for. Default is "edge_list.csv".
    
    Returns:
        nodes_df (DataFrame): Combined node data from all found node files.
        edges_df (DataFrame): Combined edge data from all found edge files.
    '''
    nodes_df = pd.DataFrame()
    edges_df = pd.DataFrame()

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)

        if os.path.isdir(full_path):
            # Recursively call the function for subfolders
            sub_nodes_df, sub_edges_df = concat_files_subfolder(full_path, node_file_name, edge_file_name)
            nodes_df = pd.concat([nodes_df, sub_nodes_df], ignore_index=True)
            edges_df = pd.concat([edges_df, sub_edges_df], ignore_index=True)
        
        elif os.path.isfile(full_path):
            if entry == node_file_name:
                # Collect node data
                print(f"Collecting nodes from: {full_path}")
                nodes_df = pd.concat([nodes_df, pd.read_csv(full_path, low_memory=False)], ignore_index=True)
            elif entry == edge_file_name:
                # Collect edge data
                print(f"Collecting edges from: {full_path}")
                edges_df = pd.concat([edges_df, pd.read_csv(full_path, low_memory=False)], ignore_index=True)

    return nodes_df, edges_df

# Adjust node and edge data (remove duplicates, rename columns)
def adjust_dataframe(node_df, edge_df):
    '''
    Adjusts the node and edge data by removing duplicates and renaming columns.
    
    Parameters:
        node_df (DataFrame): Dataframe containing node data.
        edge_df (DataFrame): Dataframe containing edge data.
    
    Returns:
        node_df (DataFrame): Adjusted node dataframe with renamed columns.
        edge_df (DataFrame): Adjusted edge dataframe with renamed columns.
    '''
    # Remove duplicate edges and reset the index
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    # Rename edge columns
    edge_df = edge_df.rename(columns={"u": "node_1", "v": "node_2"})
    # Remove duplicate nodes and reset the index
    node_df = node_df.drop_duplicates().reset_index(drop=True)
    # Rename node columns
    node_df = node_df.rename(columns={"osmid": "node_id"})
    return node_df, edge_df

# Get the current script path and add "Road_Networks"
script_path = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(script_path, "Road_Networks")
print("Current Path is:", script_path)

# List of states
state_list = ["DE", "IA", "IL", "LA", "MA", "MD", "MN", "MT", "NV", "NY"]

# Dictionary of keys and subfolders
subfolder_names = {
    "cities": "-cities-street_networks-node_edge_lists",
    "counties": "-counties-street_networks-node_edge_lists",
    "neighborhoods": "-neighborhoods-street_networks-node_edge_lists",
    "tracts": "-tracts-street_networks-node_edge_lists",
    "urbanized_areas": "-urbanized_areas-street_networks-node_edge_lists"
}

# Iterate through each state's folder, combine, and save node and edge data
for state in state_list:
    print(f"\nCollecting Data from: {state}")
    
    for key, val in subfolder_names.items():
        print(f"\n-- Collecting Nodes/Edges from: {key}")
        
        # Define paths for source and storage directories
        source_path = os.path.join(script_path, state, "Harvard Dataverse", f"{state}{val}")
        storage_path = os.path.join(script_path, state, "Road_Network_Level")
        
        # Get node and edge data from subfolders
        nodes_df, edges_df = concat_files_subfolder(source_path)
        
        # Adjust the data (remove duplicates, rename columns)
        nodes_df, edges_df = adjust_dataframe(nodes_df, edges_df)
        
        # Save node and edge data
        print(f"\n---- Storing Nodes/Edges from: {key}")
        nodes_df.to_csv(os.path.join(storage_path, "Nodes", f"nodes_{key}.csv"), index=False)
        edges_df.to_csv(os.path.join(storage_path, "Edges", f"edges_{key}.csv"), index=False)
    
    print(f"\nAppend all files for {state}")

    print("\n-- Storing Nodes (Duplicates Removed)")
    node_source_path = os.path.join(script_path, state, "Road_Network_Level", "Nodes")
    node_target_file = os.path.join(script_path, state, f"Road_Network_Nodes_{state}.csv")
    
    # Combine all node files
    concat_files(node_source_path, node_target_file)
    df_nodes = pd.read_csv(node_target_file, low_memory=False)
    
    # Remove duplicates based on node_id and keep the last occurrence
    df_nodes = df_nodes.drop_duplicates(["node_id"], keep="last")[["node_id", "x", "y"]]
    df_nodes.to_csv(node_target_file, index=False)

    print("\n-- Storing Edges (Duplicates Removed)")
    edge_source_path = os.path.join(script_path, state, "Road_Network_Level", "Edges")
    edge_target_file = os.path.join(script_path, state, f"Road_Network_Edges_{state}.csv")
    
    # Combine all edge files
    concat_files(edge_source_path, edge_target_file)
    df_edges = pd.read_csv(edge_target_file, low_memory=False)
    
    # Remove duplicates based on node_1 and node_2 and keep the last occurrence
    df_edges = df_edges.drop_duplicates(["node_1", "node_2"], keep="last")[["node_1", "node_2", "oneway", "highway", "name", "length"]]
    df_edges.to_csv(edge_target_file, index=False)