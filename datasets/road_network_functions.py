
import pandas as pd
import numpy as np
import os
from os.path import dirname



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
            df = pd.concat([df,pd.read_csv(path + file_name, low_memory=False)])
        except:
            df = pd.read_csv(path + file_name, low_memory=False)

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(final_file_name,index=False)



def concat_files_one_subfolder(path):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''


    for folder in os.listdir(path):
        try:
            edge_df = pd.concat([edge_df,pd.read_csv(path + folder + "/edge_list.csv", low_memory=False)])
        except:
            edge_df = pd.read_csv(path + folder + "/edge_list.csv", low_memory=False)

    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    edge_df = edge_df.rename(columns={"u":"node_1","v":"node_2"})

    for folder in os.listdir(path):
        try:
            node_df = pd.concat([node_df,pd.read_csv(path + folder + "/node_list.csv", low_memory=False)])
        except:
            node_df = pd.read_csv(path + folder + "/node_list.csv", low_memory=False)

    node_df = node_df.drop_duplicates().reset_index(drop=True)
    node_df = node_df.rename(columns={"osmid":"node_id"})

    return node_df, edge_df


def concat_files_two_subfolder(path):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''


    for folder in os.listdir(path):
        for subfolder in os.listdir(path + folder + "/"):
            try:
                edge_df = pd.concat([edge_df,pd.read_csv(path + folder + "/" + subfolder + "/edge_list.csv", low_memory=False)])
            except:
                edge_df = pd.read_csv(path + folder + "/" + subfolder + "/edge_list.csv", low_memory=False)

    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    edge_df = edge_df.rename(columns={"u":"node_1","v":"node_2"})

    for folder in os.listdir(path):
        for subfolder in os.listdir(path + folder + "/"):
            try:
                node_df = pd.concat([node_df,pd.read_csv(path + folder + "/" + subfolder + "/node_list.csv", low_memory=False)])
            except:
                node_df = pd.read_csv(path + folder + "/" + subfolder + "/node_list.csv", low_memory=False)

    node_df = node_df.drop_duplicates().reset_index(drop=True)
    node_df = node_df.rename(columns={"osmid":"node_id"})

    return node_df, edge_df























