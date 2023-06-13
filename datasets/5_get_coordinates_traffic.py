
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

from zipfile import ZipFile
from lxml import etree
import tabula


#--------------- Initializing Paramaters ----------#

path = dirname(os.getcwd())

state_name = "DE"



#------------------- Functions ------------------------#


def get_df(string):

    data = {
    'Attribute': [],
    'Value': []
    }

    # Extract attribute-value pairs
    lines = string.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('<th>'):
            attribute = line.replace('<th>', '').replace('</th>', '')
            data['Attribute'].append(attribute)
        elif line.startswith('<td>'):
            value = line.replace('<td>', '').replace('</td>', '')
            data['Value'].append(value)

    # Create a dataframe
    df = pd.DataFrame(data).T

    # Use the first row as column headers
    df.columns = df.iloc[0]
    df = df[1:]

    # print(df)

    return df



def extract_table_kmz(path):

    kmz = ZipFile(path, 'r')
    kml = kmz.open('doc.kml', 'r').read()

    doc = etree.fromstring(kml)

    # Iterate over elements and extract IDs
    id_list = []
    for element in doc.iter():
        element_id = element.get('id')
        if element_id:
            id_list.append(element_id)

    final_df = pd.DataFrame(columns = ['CURRENT_YEAR', 'ROAD', 'BEG_MP', 'END_MP', 'ROAD_NAME',
        'BEG_BREAKPOINT_ID', 'CURRENT_AADT', 'YEAR_LAST_COUNTED'])


    for element_id in tqdm(id_list):

        # print("\n",element_id)

        # element_id = 'kml_3418'  # ID of the element you want to extract strings from

        # Find the element by ID
        element = doc.find('.//*[@id="{}"]'.format(element_id))

        if element is not None:
            # print("Element:", element.tag)
            # print("Strings:")
            for child in element.iterchildren():
                # print("***",child.text)
                if child.text and child.text.strip() and ("<table>" in child.text):
                    # print("YESS")
                    df = get_df(child.text.strip())
                    final_df = pd.concat([final_df,df])
        else:
            print("\tElement not found.")


    return final_df





def extract_table_pdf(path_pdf):

    tables = tabula.read_pdf(path_pdf, pages="all")

    final_df = pd.DataFrame(columns = ['Maint_Rd_Number', 'Road_Name', 'End of Section Mileage', 
                                    'BEG_BREAKPNT_ID','AADT','Year Last Counted','Traffic Group'])


    for i in range(len(tables)):
        df = tables[i]

        try:
            try:
                df.columns = final_df.columns
            except:
                df = df.drop(df.columns[2],axis=1)
                df.columns = final_df.columns

            final_df = pd.concat([final_df,df])
        except:
            pass


    final_df['AADT'] = pd.to_numeric(final_df['AADT'], errors='coerce')
    final_df['End of Section Mileage'] = pd.to_numeric(final_df['End of Section Mileage'], errors='coerce')

    final_df = final_df[final_df["AADT"].isnull() == False]
    final_df = final_df[final_df["End of Section Mileage"].isnull() == False]
    final_df = final_df.drop_duplicates().reset_index(drop=True)

    return final_df



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


#----------------------- Generate Files ------------------------#

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

    final_df = pd.DataFrame(columns = ["Year",'Maint_Rd_Number', 'Road_Name', 'End of Section Mileage', 
                                    'BEG_BREAKPNT_ID','AADT','Year Last Counted','Traffic Group'])


    for year in [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
        print(year)

        for file_name in os.listdir(path + "/Traffic_Volume/" + state_name +  "/" + str(year) + "/"):

            df = extract_table_pdf(path + "/Traffic_Volume/" + state_name +  "/" + str(year) + "/" + file_name)
            df["Year"] = year
            
            final_df = pd.concat([final_df,df])


    final_df = final_df.drop_duplicates().reset_index(drop=True)

    final_df.to_csv(path + "/Traffic_Volume/" + state_name + "/" + state_name + "_AADT.csv")



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





concat_files(path + "/" + "Traffic_Volume/" + state_name + "/Coordinates/", path + "/" + "Traffic_Volume/" + state_name + "/" + state_name + "_Traffic_Volume.csv")


