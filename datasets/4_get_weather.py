from meteostat import Point, Stations, Daily, Monthly
from datetime import datetime

import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os.path import dirname


#------------- Functions ----------------------#


def get_weather(latitude, longitude, start, end):
    '''
    Retrieves weather data for a specific location and time range.

    Args:
        latitude (float): Latitude coordinate of the location.
        longitude (float): Longitude coordinate of the location.
        start (str): Start date of the time range (format: "YYYY-MM-DD").
        end (str): End date of the time range (format: "YYYY-MM-DD").

    Returns:
        pandas.DataFrame: Weather data for the specified location and time range.

    '''

    # Find the nearest weather station to the coordinate
    stations = Stations()
    station = stations.nearby(latitude, longitude).fetch(1)

    # Get the weather data for the specified station
    data = Monthly(station.index[0], start, end)
    data = data.fetch()

    # Drop the "tsun" column
    data = data.drop(columns=["tsun"])

    # Reset the index of the data
    data = data.reset_index()

    # Create a date range dataframe for the specified time range
    date_range = pd.DataFrame({"time": pd.date_range(start=start, end=end, freq='M')})
    date_range['time'] = date_range['time'].dt.to_period('M').dt.to_timestamp()

    # Merge the date range dataframe with the weather data on the "time" column
    data = pd.merge(date_range, data, on=["time"], how="left")

    return data





def concat_files(path, final_file_name):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''
    count = 0
    for file_name in tqdm(os.listdir(path)):
        try:
            df = pd.concat([df,pd.read_csv(path + file_name, low_memory=False)])
        except:
            df = pd.read_csv(path + file_name, low_memory=False)

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    # df = df.rename(columns={"x":"lat","y":"lon","X":"lat","Y":"lon"})
    df.to_csv(final_file_name,index=False)


#------------- Extract Historical Weather Data ------------------#

path = dirname(os.getcwd())

state_name = "MA"

nodes_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv", low_memory=False)

# Set time period
start = datetime(2015, 1, 1)
end = datetime(2023, 5, 1)


# Extract historical information for each coordinate

print(int(nodes_df.shape[0]/10000))
j=0
while j < int(nodes_df.shape[0]/10000):

    print(f"**** {j} ****")
    for i in tqdm(range(j*10000, (j+1)*10000)):

        latitude = nodes_df["y"][i]
        longitude = nodes_df["x"][i]
        node_id = nodes_df["node_id"][i]

        dummy = get_weather(latitude, longitude, start, end)

        dummy["node_id"] = node_id
        dummy["x"] = longitude
        dummy["y"] = latitude

        try:
            df = pd.concat([df,dummy])
        except:
            df = dummy.copy()

        # time.sleep(0.2)

    # Save the file
    df.to_csv(path + "/Weather_Features/" + state_name + "/Temp/" + state_name + "_Weather_Features" + "_" + str(j) + ".csv",index=False)

    j+=1

    time.sleep(5)


for i in tqdm(range(j*10000,df.shape[0])):

    latitude = nodes_df["y"][i]
    longitude = nodes_df["x"][i]
    node_id = nodes_df["node_id"][i]

    dummy = get_weather(latitude, longitude, start, end)

    dummy["node_id"] = node_id
    dummy["x"] = longitude
    dummy["y"] = latitude

    try:
        df = pd.concat([df,dummy])
    except:
        df = dummy.copy()

    time.sleep(0.2)


# Save the file
df.to_csv(path + "/Weather_Features/" + state_name + "/Temp/" + state_name + "_Weather_Features" + "_" + str(j) + ".csv",index=False)


concat_files(path + "/Weather_Features/" + state_name + "/Temp/", path + "/Weather_Features/" + state_name + "/" + state_name + "_Weather_Features.csv")



