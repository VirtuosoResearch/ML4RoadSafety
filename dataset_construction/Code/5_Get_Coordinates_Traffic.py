
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os

import torch
from os.path import dirname


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from tqdm import tqdm

#--------------- Initializing Paramaters ----------#

path = dirname(os.getcwd())

state_name = "DE"


def click_first_result(driver, address):

    wait = WebDriverWait(driver, 5)

    # Load the Google Maps page
    driver.get('https://www.google.com/maps')

    wait.until(EC.element_to_be_clickable((By.ID, "searchboxinput"))).send_keys(address)
    time.sleep(3)
    wait.until(EC.element_to_be_clickable((By.ID, "searchbox-searchbutton"))).click()
    time.sleep(2)


    # Get the updated URL
    updated_url = driver.current_url
    
    return updated_url

if(state_name == "MD"):

    df = pd.read_csv(path + "/MDOT_SHA_Annual_Average_Daily_Traffic_(AADT) (1).csv")

    # df["Address"] = df["STATION_DESC"] + ", " + df["ROAD_SECTION"] + ", " + df["ROADNAME"] + ", " + df["COUNTY_DESC"] + ", Maryland"

    df["Address"] = df["ROAD_SECTION"] + ", " + df["ROADNAME"] + ", " + df["COUNTY_DESC"] + ", Maryland"

    df = df[["Address",'AADT_2012',
    'AADT_2013',
    'AADT_2014',
    'AADT_2015',
    'AADT_2016',
    'AADT_2017',
    'AADT_2018',"AADT"]]

if(state_name == "DE"):

    df = pd.read_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT.csv")


df = df[["ROAD_TRAFFIC"]].drop_duplicates()
df = df.rename(columns={"ROAD_TRAFFIC":"Address"})

df["Address"] = df["Address"] + ", " + state_name

df["lat"] = 0
df["lon"] = 0

df = df.reset_index(drop=True)

# Set up Selenium WebDriver (Make sure you have the appropriate browser driver executable in your PATH)
driver = webdriver.Chrome()

for i in tqdm(range(1201,df.shape[0])):

    # Example usage
    address = df.loc[i,"Address"]
    # print(address)
    updated_url = click_first_result(driver, address)
    lat = float(updated_url.split("@")[-1].split(",")[0])
    lon = float(updated_url.split("@")[-1].split(",")[1])

    df.loc[i,"lat"] = lat
    df.loc[i,"lon"] = lon

    if(i%100 == 0):
        df_filter = df.iloc[(int(i/100)-1)*100:i,:]
        df_filter.to_csv(path + "/" + "Traffic_Volume/" + state_name + "/Coordinates/" + "MD_Traffic_Volume_" + str(int(i/100)) + ".csv",index=False)


df_filter = df.iloc[(int(i/100))*100:i,:]
df_filter.to_csv(path + "/" + "Traffic_Volume/" + state_name + "/Coordinates" + "/MD_Traffic_Volume_" + str(int(i/100)+1) + ".csv",index=False)



driver.quit()



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


concat_files(path + "/" + "Traffic_Volume/" + state_name + "/Coordinates/", path + "/" + "Traffic_Volume/" + state_name + "/" + state_name + "_Traffic_Volume.csv")


