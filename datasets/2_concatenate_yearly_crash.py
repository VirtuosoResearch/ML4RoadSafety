
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os.path import dirname

#--------------- Initializing Parameters ----------#

path = dirname(os.getcwd())


#---------------- Concatenate Yearly Files ------------#


def concat_crash_files(path, final_file_name):
    '''
    Concatenates multiple crash files into a single file.

    Args:
        path (str): Path to the directory containing crash files.
        final_file_name (str): Name of the final file to be created.

    '''

    count = 0
    file_list = os.listdir(path)
    file_list = [x for x in file_list if (x.endswith(".csv") or x.endswith(".CSV"))]

    for file_name in file_list:
        print(file_name)
        df = pd.read_csv(path + file_name, low_memory=False)

        # Convert column names to lowercase
        df.columns = map(str.lower, df.columns)

        # Process different date columns and create "accident_date" column
        if "crashyr" in df.columns:
            df['accident_date'] = pd.to_datetime(df['crashyr'].astype(str) + '-' + df['crashmonth'].astype(str) + '-' + df['crashday'].astype(str), format='%y-%m-%d')
        if "crash_month" in df.columns:
            df['accident_date'] = pd.to_datetime(df['crash_year'].astype(str) + '-' + df['crash_month'].astype(str), format='%Y-%B')
        if "acc_date" in df.columns:
            df["accident_date"] = pd.to_datetime(df["acc_date"], format="%Y%m%d")

        # Rename latitude and longitude columns
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Rename various date columns to "accident_date"
        df = df.rename(columns={"crash datetime": "accident_date", "date of crash": "accident_date", "crash date": "accident_date", "crash_date": "accident_date"})

        # Rename columns "x" and "y" to "lat" and "lon"
        if "lat" not in df.columns:
            df = df.rename(columns={"x": "lat", "y": "lon"})

        # Add "acc_count" column and group by "accident_date", "lat", "lon"
        df["acc_count"] = 1
        df = df.groupby(["accident_date", "lat", "lon"], as_index=False)["acc_count"].sum()

        # Remove rows with latitude or longitude of 0
        df = df[(df["lat"] != 0) | (df["lon"] != 0)]

        # Drop duplicate rows and rows with missing values
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df = df.astype(str)

        if count > 0:
            # Concatenate with the previous dataframe
            prev_df = pd.read_pickle(final_file_name)
            prev_df = prev_df.astype(str)
            df = pd.concat([df, prev_df], axis=0, ignore_index=True)

        count += 1

        # Remove duplicate rows again
        df = df.drop_duplicates().reset_index(drop=True)

        # Save the dataframe as a pickle file
        df.to_pickle(final_file_name, protocol=5)
     

for state_name in ["MA"]:

    print(f"\n********* {state_name} ***********")

    concat_crash_files(path + "/Accidents/" + state_name + "/Crashes_Year/", path + "/Accidents/" + state_name + "/" + state_name + "_crash.pkl")


