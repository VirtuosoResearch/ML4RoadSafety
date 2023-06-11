#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os.path import dirname
from zipfile import ZipFile
from lxml import etree
import tabula

#--------------- Initializing Parameters ----------#

path = dirname(os.getcwd())

state_name = "DE"



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


final_df = pd.DataFrame(columns = ["Year",'Maint_Rd_Number', 'Road_Name', 'End of Section Mileage', 
                                    'BEG_BREAKPNT_ID','AADT','Year Last Counted','Traffic Group'])


for year in [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
    print(year)

    for file_name in os.listdir(path + "/Traffic_Volume/" + state_name +  "/" + str(year) + "/"):

        df = extract_table_pdf(path + "/Traffic_Volume/" + state_name +  "/" + str(year) + "/" + file_name)
        df["Year"] = year
        
        final_df = pd.concat([final_df,df])


final_df = final_df.drop_duplicates().reset_index(drop=True)











