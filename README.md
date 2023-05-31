# Graph Learning for Road Safety

## Data

We have collected/extracted the following data to create the graph for each state:
- Road (Street) Networks
- Accident Records
- Traffic Volume
- Weather Features
- Inherent Graph Features




[Download link](https://drive.google.com/drive/folders/1PHIkgoKkugj6rMvkbjxpJbmzVn69dm8e)


### Road Network / Street Network:

The road network is created as a graph where the nodes and edges are defined as below:

- **Node:**
Intersection of roads (latitude and longitude)
- **Edge:**
Road (length, name, type of road, etc)

There are two methods to extract the desired network:

- **OSMnx library:**
  
  The library “osmnx” from OpenStreetMaps uses an api to extract the necessary features. For this project, since the street network is required, the api takes a bounding box of coordinates as an input and extracts the street network as an output.
  
  Since the project requires the street network of an entire city/state, the geographic region is split into multiple small rectangles and the street network is extracted for each one of them, and finally appended into one big network.

- **[Harvard Dataverse](https://doi.org/10.7910/DVN/CUWWYJ):**
  
  The required street network has been published in Harvard Dataverse in 2017. For every state, the street networks are available at the scale of city, county, neighbourhood, tract and urbanized area.

  The street networks for all of the above mentioned levels of a state are appended to ensure all the streets in the state are included in the road network graph for that state. 

  The road network is available for all the states in United States of America.



### Accident Records:

Accident records have been obtained for states where the data is available at a person/vehicle level and the lat-lon coordinates are available or can be extracted.

The accident records for all such states have been obtained from the data published by the Department of Transportation for the respective state.

Here is a summary of the records collected:

1. **[Delaware (DE)](https://data.delaware.gov/Transportation/Public-Crash-Data-Map/3rrv-8pfj):** 458,282 accident records from Jan 1 2009 to Oct 31 2022

2. **[Iowa (IA)](https://icat.iowadot.gov/#):** 556,418 accident records from Jan 1 2013 to May 1 2023


3. **[Illinois (IL)](https://gis-idot.opendata.arcgis.com/search?collection=Dataset&q=Crashes):** 2,980,702 accident records from Jan 1 2012 to Dec 31 2021

4. **[Maryland (MD)](https://opendata.maryland.gov/Public-Safety/Maryland-Statewide-Vehicle-Crashes/65du-s3qu):** 878,343 accident records from Jan 1 2015 to Dec 31 2022

5. **[Massachusetts (MA)]():** 3,296,566 accident records from Jan 1 2002 to Dec 31 2022 except the years 2018, 2019

6. **[Minnesota (MN)](https://mncrash.state.mn.us/Pages/AdHocSearch.aspx):** 513,969 accident records from Dec 31 2015 to May 1 2023

7. **[Montana (MT)](https://www.mdt.mt.gov/publications/datastats/crashdata.aspx):** 99,939 accident records from Jan 2016 to Dec 2020

8. **[Nevada (NV)](https://ndot.maps.arcgis.com/apps/webappviewer/index.html?id=00d23dc547eb4382bef9beabe07eaefd):** 237,338 accident records from Jan 1 2016 to Dec 31 2020

9. **[Vermont (VT)]():** 

10. **New York City:** A dataset that contains information about road accidents in New York City from April 2014 to March 2023. The dataset contains 1,000,000+ accidents.



### Traffic Volume:

The Traffic data is extracted from the data published by the Department of Transportation (DOT) of every state.


1. **[Delaware (DE)](https://deldot.gov/search/):** Data available from 2009 to 2019 in pdfs and .kmz files

2. **[Iowa (IA)]():** 

3. **[Illinois (IL)]():** 

4. **[Maryland (MD)]():** 

5. **[Massachusetts (MA)](https://mhd.public.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=):** 
  
    Traffic counts at 9,444 locations. Only the latest data is available with the corresponding year. The yearly growth_rate for different road types has been taken from [here](https://www.mass.gov/lists/massdot-historical-traffic-volume-data), which is averaged over the years for every road type. The historical data is then extrapolated for all the years.

6. **[Minnesota (MN)]():** 

7. **[Montana (MT)]():** 

8. **[Nevada (NV)]():** 

9.  **[Vermont (VT)]():** 

10. **[New York City (NYC)](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt):**
New York City Department of Transportation (NYC DOT) uses Automated Traffic Recorders (ATR) to collect traffic sample volume counts at bridge crossings and roadways.These counts do not cover the entire year, and the number of days counted per location may vary from year to year.
The dataset contains traffic counts as defined above from 2011 to 2019.


### Weather:

The weather data is extracted using meteostat api. For every node (intersection) in the state, the historical weather data is extracted corresponding to the data recorded at the nearest station to that node. Since this data is temporal in nature and is available at every node, we believe that the weather features would be important for our task.

The following features are extracted at a Month level:

- tavg: Average Surface Temperature
- tmin: Minimum Surface Temperature
- tmax: Maximum Surface Temperature
- prcp: Total Precipitation
- wspd: Avg Wind Speed
- pres: Sea Level Air Pressure


After extracting the historical data for a particular node, the feature data for a few months is not available. The following methodology has been adopted to impute this missing data:

  - Replace with the feature data of the same month the following year
  - Replace with the feature data of the same month the previous year
  - Replace with the average of previous month and next month feature data


### Inherent Graph Features:

The following features have been calculated which might help in improving the performance of the model:

- Node Degree
- Betweenness Centrality
- Node Position
  - Latitude
  - Longitude



### Creating the Final dataset:

All the above datasets need to be combined in order to create the final graph for modeling. 

- **Accident Data, Road Network:**
    - The accident data contains the lat-long coordinates of every accident which has to be mapped to the nearest street in the data generated from OpenStreetMaps. 
    - The road network has the lat-long coordinates of every intersection/node in the graph. 
    - We assume that the accident takes place at some point between two nodes. 
    - Let’s assume the accident takes place at point C which is on the street between nodes A and B. 
    - The distance AC + BC should ideally be equal to AB assuming the street AB is a straight road. 
    - Using the above methodology, we iterate over all the streets and map the accident to the street where the difference of distances (AB - (AC+BC)) is the lowest.

- **Traffic Data, Road Network:**
    - Traffic Data contains the names of the streets defined by Department of Transportation. Road Network has the names of streets given by OpenStreetMaps. These names have to be mapped to combine both these data sources
    - Extract the lat-lon coordinates from the road names in the traffic data
    - Apply the nearest street mapping methodology described above to map the street where the traffic was recorded to the edge 
    - Extract the county name from the lat-lon coordinate
    - Calculate the AADT value of a particular county by averaging the AADT values of all the streets in that county
  - Assign this value to all the streets in that county
  - Repeat the above steps for all counties, finally getting the AADT values for all streets in the state
  
- **Weather Data, Road Network:**

  - From the lat lon coordinate of a partular node, extract the historical weather features using the methodology explained above.
  - Repeat this exercise for all nodes


## Features

### Node Features

These are the following node features:

- Latitude
- Longitude
- Node Degree
- Betweenness Centrality
- Average Surface Temperature
- Max Surface Temperature
- Min Surface Temperature
- Total Precipitation
- Avg Wind Speed
- Sea Level Air Pressure


### Edge Features

These are the following edge features:

- oneway
- highway
- length
- Annual Average Daily Traffic (AADT)

## Baseline Models

Now that we have created the dataset/graph, we need to model Graph Neural Networks (GNNs) for our prediction task. Since the task is to predict if an accident has taken place on a street or not, it is an edge classification task in our graph.

We ran experiments on several baseline models, including:

- **[DCRNN](https://arxiv.org/abs/1707.01926):** A graph neural network that consists of a diffusion convolutional layer to capture spatial dependencies, a recurrent layer to capture temporal dependencies, and a pooling layer to aggregate information across multiple time steps.

- **[Graph-WaveNet](https://arxiv.org/abs/1906.00121):** A graph convolutional neural network that utlizes a self-adaptive adjacency matrix to capture spatial-temporal dependencies simultaneously.

- **[STGCN](https://arxiv.org/abs/1709.04875):** A graph convolutional neural network that uses a spatio-temporal graph convolutional network architecture to aggregate information across multiple time steps in traffic forecasting tasks.

## Installation

To install the required packages, run the following command:
    
    pip install -r requirements.txt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
