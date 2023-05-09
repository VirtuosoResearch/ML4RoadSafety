# Learning on Traffic Networks for Road Safety

We present a methodology for predicting road accidents on street networks using Graph Neural Networks (GNNs). Our method defines the edges of the network as the streets and the nodes as the intersections, allowing us to model the complex dependencies among road segments and their surrounding areas. To test our approach, we gathered traffic and accident datasets from various states across the United States through public sources and aligned each record with the corresponding edge of the network to set up the prediction problem.

Our primary objective is to anticipate the likelihood of an accident occurring at a given road segment, given past road conditions. Additionally, we address the spatial-temporal challenge of predicting traffic volume on the street network. Our GNN-based approach shows promising results demonstrating its potential to improve road safety measures and traffic management systems.

We believe our approach can have practical applications in urban planning, transportation engineering, and public safety, and we hope that our results can pave the way for further research in this area.

## Introduction

Road accidents not only cause immense human suffering but also have a significant economic impact, with billions of dollars lost each year due to medical expenses, lost productivity, and property damage. To address this challenge, several states have adopted the Vision Zero agenda, a goal to eliminate traffic fatalities and serious injuries. Machine learning techniques such as Graph Neural Networks (GNNs) can play a vital role in achieving this objective by helping to predict the likelihood of accidents and prevent them from happening.

In this project, we utilize GNNs for predicting road accidents on street networks. By defining the edges of the network as the streets and the nodes as the intersections, we model the complex dependencies among road segments and their surrounding areas, enabling us to make accurate predictions. We align traffic and accident datasets from various states across the United States to the corresponding edge of the network to set up the prediction problem and train our model to anticipate the likelihood of an accident at a given road segment, based on past road conditions.

Our approach not only addresses the practical challenge of predicting accidents but also contributes to the Vision Zero agenda of these states. By providing insights and actionable information to city planners and policymakers, our approach helps them to make informed decisions to reduce the likelihood of accidents and improve road safety. Moreover, our approach also addresses the spatial-temporal challenge of predicting traffic volume on the street network, which is critical in developing effective traffic management strategies.

Our results demonstrate the potential of GNNs in predicting accidents and traffic volume, providing a promising avenue for improving public safety and achieving the Vision Zero agenda. We believe that our approach can have practical applications in urban planning, transportation engineering, and public safety, and we hope that our results can inspire further research and development in this area.

## Data

We have collected 3 types of data to create the graph:
- Road Network / Street Network
- Accident Records
- Traffic Volume


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

1. **[Delaware (DE)](https://data.delaware.gov/Transportation/Public-Crash-Data-Map/3rrv-8pfj):** 458,285 accident records from Jan 1 2009 to Oct 31 2022

2. **[Iowa (IA)](https://icat.iowadot.gov/#):** 556,698 accident records from Jan 1 2013 to May 1 2023


3. **[Illinois (IL)](https://gis-idot.opendata.arcgis.com/search?collection=Dataset&q=Crashes):** 2,980,702 accident records from Jan 1 2012 to Dec 31 2021

4. **[Maryland (MD)](https://opendata.maryland.gov/Public-Safety/Maryland-Statewide-Vehicle-Crashes/65du-s3qu):** 878,343 accident records from Jan 1 2015 to Dec 31 2022

5. **Massachusetts (MA):** 

6. **[Minnesota (MN)](https://mncrash.state.mn.us/Pages/AdHocSearch.aspx):** 514,542 accident records from Dec 31 2015 to May 1 2023

7. **[Montana (MT)](https://www.mdt.mt.gov/publications/datastats/crashdata.aspx):** 99,934 accident records from Jan 2016 to Dec 2020

8. **[Nevada (NV)](https://ndot.maps.arcgis.com/apps/webappviewer/index.html?id=00d23dc547eb4382bef9beabe07eaefd):** 237,388 accident records from Jan 1 2016 to Dec 31 2020

9. **[Vermont (VT)]():** 

10. **New York City:** A dataset that contains information about road accidents in New York City from April 2014 to March 2023. The dataset contains 1,000,000+ accidents.



### Traffic Volume:

The Traffic data is extracted from the data published by the Department of Transportation (DOT) of every state.

- **NYC [(Link)](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt):**
New York City Department of Transportation (NYC DOT) uses Automated Traffic Recorders (ATR) to collect traffic sample volume counts at bridge crossings and roadways.These counts do not cover the entire year, and the number of days counted per location may vary from year to year.
The dataset contains traffic counts as defined above from 2011 to 2019.



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
    - Traffic Data contains the names of the streets defined by Department of Transportation
    - Road Network has the names of streets given by OpenStreetMaps
    - The names have to be manually mapped to combine both these data sources


## Features

### Node Features

There are 2 node features:

- Latitude
- Longitude


### Edge Features

There are - edge features:

- oneway
- highway
- length

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
