# Data Collection Process

We document our data collection process using publicly available sources, including:
- Traffic accident records
- Road networks
- Road network features, including traffic volume reports, weather conditions, and other graph structural features

To reproduce the data for a state, download the data for each feature and save these in the same local repository. We describe the steps needed to generate the processed graphs, also giving instructions on how to run the code for MA, which can be replicated for other states.

### Constructing road networks

The road network is created as a graph where the nodes and edges are defined as below:

- **Node:**
Intersection of roads (latitude and longitude)
- **Edge:**
Road (length, name, type of road, etc)

- **OSMnx Street Network Dataverse**
  
  The required street network has been published in OSMnx Street Network Dataverse. For every state, the street networks are available at the scale of city, county, neighbourhood, tract and urbanized area. The street networks for all of the above mentioned levels of a state are appended to ensure all the streets in the state are included in the road network graph for that state. 

**Example: Constructing the road network of MA:** Download all the node_edge_lists zip files for MA from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CUWWYJ), and run `1_road_network.py` to get the final road network for MA.

```bash
mkdir ./Road_Networks
mkdir ./Road_Network_Level
cd ./Road_Networks
mkdir ./MA
cd ./MA
mkdir ./Harvard Dataverse
mkdir ./Road_Network_Level
# move downloaded dataset in `./Harvard Dataverse`
```


### Collecting traffic accident records

Accident records have been obtained for states where the data is available at a person/vehicle level and the lat-lon coordinates are available or can be extracted. The accident records for all such states have been obtained from the data published by the Department of Transportation for the respective state. 

Here is a summary of the records collected for every state:

1. Delaware (DE): 458,282 accident records from Jan 1 2009 to Oct 31 2022

2. Iowa (IA): 556,418 accident records from Jan 1 2013 to May 1 2023

3. Illinois (IL): 2,980,702 accident records from Jan 1 2012 to Dec 31 2021

4. Maryland (MD): 878,343 accident records from Jan 1 2015 to Dec 31 2022

5. Massachusetts (MA): 3,296,566 accident records from Jan 1 2002 to Dec 31 2022 except the years 2018, 2019

6. Minnesota (MN): 513,969 accident records from Dec 31 2015 to May 1 2023

7. Montana (MT): 99,939 accident records from Jan 2016 to Dec 2020

8. Nevada (NV): 237,338 accident records from Jan 1 2016 to Dec 31 2020

9. New York City (NYC): 1,817,820 accident records from Jul 1 2012 to Feb 12 2024

10. Los Angeles County (LA): 603,629 accident records from Jan 1 2010 to Feb 12 2024

**Example: Processing accident records from MA:** Download the yearwise crash reports of MA from [here](https://geo-massdot.opendata.arcgis.com/search?collection=Dataset&q=crash), and run `2_concatenate_yearly_crash.py` and `3_extract_nearest_street.py` to get the processed accident records for MA.

```bash
mkdir ./Accidents
cd ./Accidents
mkdir ./MA
cd ./MA
mkdir ./Crashes_Year
mkdir ./Nearest_Street
cd ./Crashes_Year
# move downloaded data into Crashes_Year and rename in this format: `2002_Vehicle_Level_Crash_Details.csv`
```

### Collecting road network features

The weather data is extracted using meteostat api. For every node (intersection) in the state, the historical weather data is extracted corresponding to the data recorded at the nearest station to that node.  

**Example: Processing road network features for MA:** Run `4_get_weather.py` to extract the historical weather data for all nodes in MA.

```bash
mkdir ./Weather_Features
cd ./Weather_Features
mkdir ./MA
cd ./MA
mkdir ./Temp
cd ../../
```


The traffic volume data is extracted from the data published by the Department of Transportation (DOT) of every state and is measured by Annual Average Daily Traffic (AADT). Here is a summary of the records collected for every state:

1. Delaware (DE): Data available in pdfs and .kmz files at a road level. The corresponding coordinates have been extracted using google maps api.

2. Maryland (MD): Road level data available at a historical level. The corresponding coordinates have been extracted using google maps api.

3. Massachusetts (MA): Historical data available at a coordinate level.

4. Nevada (NV): Historical data available at a coordinate level.

5. New York City (NYC): Historical data available at a coordinate level.


**Example: Processing traffic volume records of MA:** Download yearly historical traffic counts from [here](https://mhd.public.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=), and run `5_get_coordinates_traffic.py` and `6_get_traffic_volume.py` to get the processed AADT counts.

Besides, the following structural features have been calculated which would help in improving the performance of the model: Node degree, betweenness centrality, and node position (latitude and longitude).

### Alignment of network labels and features 

Lastly, to generate the final graphs and labels, we need to map all the above generated features to the road network of that state and process it in the necessary format for modelling.

**Example: Mapping accidents to the road network of MA:** Run `7_dataset_creation.py` to get the final processed graphs as deposited [here](https://dataverse.harvard.edu/privateurl.xhtml?token=add1d658-0e71-4007-9735-7976efb8de5e).

### Conclusion

Finally, the processed dataset is stored in a structure as follows. For example, for the state of MA, the directory looks like the following:

```
--MA/                         # take the Massachusetts state for an example
  |
  ---- adj_matrix.pt          # the sparse adjacency matrix of the road network
  |
  ---- accidents_monthly.csv  # all accidents spanning multiple years and aggregated in months
  |
  ---- Nodes/                 # the node features, including weather information every month
  |     |
  |     ---- node_features_{year}_{month}.pt # weather information of a particular month
  |     |
  |     ---- ...
  |
  ---- Edges/                 # the edge features, including road and traffic volume information if available
        |
        ---- edge_features.pt                # edge features describing the road information
        |
        ---- edge_features_traffic_{year}.pt # traffic volume records of a particular year
        |
        ---- ...
```

Find the processed files at this [repository](https://dataverse.harvard.edu/privateurl.xhtml?token=add1d658-0e71-4007-9735-7976efb8de5e)!
