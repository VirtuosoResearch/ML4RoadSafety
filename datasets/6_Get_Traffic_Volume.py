#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os.path import dirname

#--------------- Initializing Parameters ----------#

path = dirname(os.getcwd())

state_name = "DE"


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
    # df = df.rename(columns={"x":"lat","y":"lon","X":"lat","Y":"lon"})
    df.to_csv(final_file_name,index=False)



if(state_name == "MA"):

    # concat_files(path + "/Traffic_Volume/" + state_name + "/County/",path + "/Traffic_Volume/" + state_name + "/tcds.csv")

    tcds_df = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/tcds.csv")
    growth_df = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/growth_rate.csv")


    tcds_df = tcds_df[~tcds_df["Latest"].isnull()].reset_index(drop=True)
    tcds_df["Group"] = tcds_df["Rural Urban"] + tcds_df["Functional Class"].apply(lambda x: x[1])

    aadt_df = pd.merge(tcds_df,growth_df,on=["Group"],how="left").drop_duplicates()
    aadt_df["Year"] = pd.to_datetime(aadt_df["Latest Date"]).dt.year

    for i in range(2002,2023):
        aadt_df["AADT_" + str(i)] = 0

    for i in tqdm(range(aadt_df.shape[0])):
        aadt = aadt_df.loc[i,"Latest"]
        year = aadt_df.loc[i,"Year"]
        growth_rate = aadt_df.loc[i,"Growth_Rate"]

        for year_est in range(2002,2023):
            aadt_df.loc[i,"AADT_"+str(year_est)] = int(aadt * ((1 + growth_rate) ** (year_est - year)))


    aadt_df = aadt_df.rename(columns={"Latitude":"lat","Longitude":"lon"})
    aadt_df["lat"] = pd.to_numeric(aadt_df["lat"])
    aadt_df["lon"] = pd.to_numeric(aadt_df["lon"])


    nodes_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv", low_memory=False)
    edges_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Edges_" + state_name + ".csv", low_memory=False)

    # Merge the latlong coordinates of the nodes of all the edges
    edges_df = pd.merge(edges_df,nodes_df,left_on="node_1",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_1_x","y":"node_1_y"})

    edges_df = pd.merge(edges_df,nodes_df,left_on="node_2",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_2_x","y":"node_2_y"})

    # Calculate the distance between 2 nodes
    edges_df["street_dist"] = np.sqrt((edges_df["node_2_x"] - edges_df["node_1_x"])**2 + (edges_df["node_2_y"] - edges_df["node_1_y"])**2)

    aadt_df["node_1"] = 0
    aadt_df["node_2"] = 0


    for i in tqdm(range(aadt_df.shape[0])):
        aadt_df.loc[i,"node_1"],aadt_df.loc[i,"node_2"] = extract_nearest_street(edges_df,aadt_df.loc[i,"lat"],aadt_df.loc[i,"lon"])

    final_df = pd.DataFrame(columns=["node_1","node_2","AADT","year"])
    for year in range(2002,2023):

        df = aadt_df[["node_1","node_2","AADT_"+str(year)]].reset_index(drop=True).drop_duplicates()
        df["year"] = year
        df = df.rename(columns={"AADT_"+str(year):"AADT"})

        final_df = pd.concat([final_df,df])

    final_df = final_df.reset_index(drop=True).drop_duplicates()

    final_df.to_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT.csv",index=False)

if(state_name == "MD"):

    aadt_df = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_Traffic_Volume.csv")

    nodes_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv", low_memory=False)
    edges_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Edges_" + state_name + ".csv", low_memory=False)

    # Merge the latlong coordinates of the nodes of all the edges
    edges_df = pd.merge(edges_df,nodes_df,left_on="node_1",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_1_x","y":"node_1_y"})

    edges_df = pd.merge(edges_df,nodes_df,left_on="node_2",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_2_x","y":"node_2_y"})

    # Calculate the distance between 2 nodes
    edges_df["street_dist"] = np.sqrt((edges_df["node_2_x"] - edges_df["node_1_x"])**2 + (edges_df["node_2_y"] - edges_df["node_1_y"])**2)


    aadt_df = aadt_df.rename(columns={"AADT":"AADT_2019"})


    aadt_df["node_1"] = 0
    aadt_df["node_2"] = 0
    for i in tqdm(range(aadt_df.shape[0])):
        aadt_df.loc[i,"node_1"],aadt_df.loc[i,"node_2"] = extract_nearest_street(edges_df,aadt_df.loc[i,"lat"],aadt_df.loc[i,"lon"])

    final_df = pd.DataFrame(columns=["node_1","node_2","AADT","year"])
    for year in range(2012,2020):

        df = aadt_df[["node_1","node_2","AADT_"+str(year)]].reset_index(drop=True).drop_duplicates()
        df["year"] = year
        df = df.rename(columns={"AADT_"+str(year):"AADT"})

        final_df = pd.concat([final_df,df])

    final_df = final_df.reset_index(drop=True).drop_duplicates()

    final_df.to_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT.csv",index=False)

if(state_name == "DE"):

    mapping_df = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_Road_Coordinate_Mapping.csv")

    nodes_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv", low_memory=False)
    edges_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Edges_" + state_name + ".csv", low_memory=False)

    # Merge the latlong coordinates of the nodes of all the edges
    edges_df = pd.merge(edges_df,nodes_df,left_on="node_1",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_1_x","y":"node_1_y"})

    edges_df = pd.merge(edges_df,nodes_df,left_on="node_2",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_2_x","y":"node_2_y"})

    # Calculate the distance between 2 nodes
    edges_df["street_dist"] = np.sqrt((edges_df["node_2_x"] - edges_df["node_1_x"])**2 + (edges_df["node_2_y"] - edges_df["node_1_y"])**2)


    mapping_df["node_1"] = 0
    mapping_df["node_2"] = 0
    for i in tqdm(range(mapping_df.shape[0])):
        mapping_df.loc[i,"node_1"],mapping_df.loc[i,"node_2"] = extract_nearest_street(edges_df,mapping_df.loc[i,"lat"],mapping_df.loc[i,"lon"])

    aadt_df = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT_unmapped.csv")

    aadt_df["ROAD_TRAFFIC"] = aadt_df["ROAD_TRAFFIC"] + ", DE"
    aadt_df = aadt_df.rename(columns={"Year":"year"})

    final_df = pd.merge(aadt_df,mapping_df,left_on=["ROAD_TRAFFIC"],right_on=["Address"],how="left")


    final_df = final_df[["node_1","node_2","AADT","year"]].reset_index(drop=True).drop_duplicates()


    final_df.to_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT.csv",index=False)




