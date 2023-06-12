# Data Collection Process

We have meticulously documented our data collection process, providing valuable insights into how we processed accidents and road networks using publicly available sources. This information serves as a reference for researchers and practitioners interested in analyzing traffic accidents.

We describe the collection procedure for each data source, including:
- Traffic accident records
- Road networks
- Road network features, including traffic volume reports, weather conditions, and other graph structural features


### Collecting traffic accident records

Accident records have been obtained for states where the data is available at a person/vehicle level and the lat-lon coordinates are available or can be extracted. The accident records for all such states have been obtained from the data published by the Department of Transportation for the respective state. Here is a summary of the records collected:

1. Delaware (DE): 458,282 accident records from Jan 1 2009 to Oct 31 2022

2. Iowa (IA): 556,418 accident records from Jan 1 2013 to May 1 2023

3. Illinois (IL): 2,980,702 accident records from Jan 1 2012 to Dec 31 2021

4. Maryland (MD): 878,343 accident records from Jan 1 2015 to Dec 31 2022

5. Massachusetts (MA): 3,296,566 accident records from Jan 1 2002 to Dec 31 2022 except the years 2018, 2019

6. Minnesota (MN): 513,969 accident records from Dec 31 2015 to May 1 2023

7. Montana (MT): 99,939 accident records from Jan 2016 to Dec 2020

8. Nevada (NV): 237,338 accident records from Jan 1 2016 to Dec 31 2020


### Constructing road networks:

The road network is created as a graph where the nodes and edges are defined as below:

- **Node:**
Intersection of roads (latitude and longitude)
- **Edge:**
Road (length, name, type of road, etc)

- **OSMnx Street Network Dataverse:**
  
  The required street network has been published in OSMnx Street Network Dataverse in 2017. For every state, the street networks are available at the scale of city, county, neighbourhood, tract and urbanized area.

  The street networks for all of the above mentioned levels of a state are appended to ensure all the streets in the state are included in the road network graph for that state. 

  The road network is available for all the states in United States of America.


### Collecting traffic volume:

The traffic volume data is extracted from the data published by the Department of Transportation (DOT) of every state.

1. Delaware (DE): Data available in pdfs and .kmz files at a road level. The corresponding coordinates have been extracted using google maps api.

2. Maryland (MD): Road level data available at a historical level. The corresponding coordinates have been extracted using google maps api.

3. Massachusetts (MA): Historical data available at a coordinate level.

4. Nevada (NV): Historical data available at a coordinate level.

### Collecting road network features:

The weather data is extracted using meteostat api. For every node (intersection) in the state, the historical weather data is extracted corresponding to the data recorded at the nearest station to that node.  

### Collecting road network features:

Besides, the following structural features have been calculated which would help in improving the performance of the model:

- Node degree
- Betweenness Centrality
- Node position: Latitude and longitude

### Alignment of network labels and features 

Lastly, we combine the above in order to create the final graphs for modeling. 

**Traffic accident records**
- The accident data contains the lat-long coordinates of every accident which has to be mapped to the nearest street in the data generated from OpenStreetMaps. 
- The road network has the lat-long coordinates of every intersection/node in the graph. 
- We assume that the accident takes place at some point between two nodes. 
- Let’s assume the accident takes place at point C which is on the street between nodes A and B. 
- The distance AC + BC should ideally be equal to AB assuming the street AB is a straight road. 
- Using the above methodology, we iterate over all the streets and map the accident to the street where the difference of distances (AB - (AC+BC)) is the lowest.

**Traffic Volume Counts**
- Traffic Data contains the names of the streets defined by Department of Transportation. Road Network has the names of streets given by OpenStreetMaps. These names have to be mapped to combine both these data sources
- Extract the lat-lon coordinates from the road names in the traffic data
- Apply the nearest street mapping methodology described above to map the street where the traffic was recorded to the edge 
- Extract the county name from the lat-lon coordinate
- Calculate the AADT value of a particular county by averaging the AADT values of all the streets in that county
- Assign this value to all the streets in that county
- Repeat the above steps for all counties, finally getting the AADT values for all streets in the state
- For weather Data, from the lat lon coordinate of a partular node, extract the historical weather features using the methodology explained above; Repeat this exercise for all nodes


## Reproduction

To reproduce the data for a state, download the data for each feature and save these in the same local repository. Run the dataset codes in the numerical order to get the processed files as deposited in https://dataverse.harvard.edu/privateurl.xhtml?token=add1d658-0e71-4007-9735-7976efb8de5e .





