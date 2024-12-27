from meteostat import Point, Stations, Monthly
from datetime import datetime

import time
import pandas as pd
import os
from tqdm import tqdm

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

    # Drop the "tsun" column if it exists
    if "tsun" in data.columns:
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
    Combines all files in a directory and saves it in a single file.
    
    Parameters:
        path (str): Directory where all independent files are saved.
        final_file_name (str): Path of the final file.
    '''
    count = 0
    for file_name in tqdm(os.listdir(path)):
        try:
            df = pd.concat([df, pd.read_csv(os.path.join(path, file_name), low_memory=False)])
        except:
            df = pd.read_csv(os.path.join(path, file_name), low_memory=False)

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(final_file_name, index=False)


#------------- Extract Historical Weather Data ------------------#
path = os.path.dirname(os.path.abspath(__file__))

# List of state names to process
state_names = ["MA"]  # "DE", "IA", "IL", "LA", "MD", "MN", "MT", "NY", "MA"

# Set time period
start = datetime(2023, 1, 1)
end = datetime(2024, 12, 1)

# Process each state in the list
for state_name in state_names:

    print(f"\n********** Processing State: {state_name} **********")

    # Read the road network nodes for the state
    nodes_df = pd.read_csv(os.path.join(path, "Road_Networks", state_name, f"Road_Network_Nodes_{state_name}.csv"), low_memory=False)

    # Process in batches
    print(f"Total batches: {int(nodes_df.shape[0] / 10000)}")
    j = 0  # Starting batch index
    while j < int(nodes_df.shape[0] / 10000):

        print(f"**** Batch {j} ****")
        for i in tqdm(range(j * 10000, (j + 1) * 10000)):

            latitude = nodes_df["y"][i]
            longitude = nodes_df["x"][i]
            node_id = nodes_df["node_id"][i]

            # Fetch weather data
            dummy = get_weather(latitude, longitude, start, end)

            # Add metadata columns
            dummy["node_id"] = node_id
            dummy["x"] = longitude
            dummy["y"] = latitude

            # Concatenate data
            try:
                df = pd.concat([df, dummy])
            except:
                df = dummy.copy()

        # Save batch data
        os.makedirs(os.path.join(path, "Weather_Features", state_name, "Temp"), exist_ok=True)
        df.to_csv(os.path.join(path, "Weather_Features", state_name, "Temp", f"{state_name}_Weather_Features_{j}.csv"), index=False)

        j += 1
        time.sleep(5)

    # Process remaining rows outside of full batches
    for i in tqdm(range(j * 10000, nodes_df.shape[0])):

        latitude = nodes_df["y"][i]
        longitude = nodes_df["x"][i]
        node_id = nodes_df["node_id"][i]

        # Fetch weather data
        dummy = get_weather(latitude, longitude, start, end)

        # Add metadata columns
        dummy["node_id"] = node_id
        dummy["x"] = longitude
        dummy["y"] = latitude

        # Concatenate data
        try:
            df = pd.concat([df, dummy])
        except:
            df = dummy.copy()

        time.sleep(0.2)

    # Save the remaining rows
    df.to_csv(os.path.join(path, "Weather_Features", state_name, "Temp", f"{state_name}_Weather_Features_{j}.csv"), index=False)

    # Combine all temporary files into a single file for the state
    concat_files(
        os.path.join(path, "Weather_Features", state_name, "Temp/"),
        os.path.join(path, "Weather_Features", state_name, f"{state_name}_Weather_Features.csv")
    )

print("\nAll states processed successfully!")