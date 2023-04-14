# Predicting Road Accidents using Graph Neural Networks

This is a machine learning project that uses Graph Neural Networks (GNNs) to predict road accidents. We have collected traffic and accident datasets from various cities across the United States from public sources. The data is from the Massachusetts Department of Transportation (MassDOT) and the New York City Police Department. The goal is to predict the likelihood of an accident at a given road segment in a given previous road conditions. 

## Introduction

Road accidents are a major cause of injury and death worldwide. Machine learning techniques can help us to predict the likelihood of accidents and prevent them from happening. 

## Datasets

We have collected 3 types of data to create the graph:
- Accident Data
- Traffic Data
- Road Network Data

[Download link](https://drive.google.com/drive/folders/1PHIkgoKkugj6rMvkbjxpJbmzVn69dm8e)

### Accident Data:

- **Massachusetts:** A dataset that contains information about road accidents in Massachusetts from January 2015 to February 2023. The dataset contains 3000+ accidents.

- **New York City:** A dataset that contains information about road accidents in New York City from April 2014 to March 2023. The dataset contains 1 Million+ accidents.

### Traffic Data:

The Traffic data is extracted from the data published by the Department of Transportation (DOT) of every state.

- **NYC [(Link)](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt):**
New York City Department of Transportation (NYC DOT) uses Automated Traffic Recorders (ATR) to collect traffic sample volume counts at bridge crossings and roadways.These counts do not cover the entire year, and the number of days counted per location may vary from year to year.
The dataset contains traffic counts as defined above from 2011 to 2019.

### Road Network Data:

We have used the library “osmnx” from OpenStreetMaps which uses an api to extract the necessary features. For this project, since the street network is required, the api takes a bounding box of coordinates as an input and extracts the street network as an output.

The street network is created as a graph where the nodes and edges are defined as below:

- **Node:**
Intersection of roads (latitude and longitude)
- **Edge:**
Road (length, name, type of road, etc)

Since the project requires the street network of an entire city/state, the geographic region is split into multiple small rectangles and the street network is extracted for each one of them, and finally appended into one big network.

### Creating the Final dataset:

All the above datasets need to be combined in order to create the final graph for modeling. 

- **Accident Data, Road Network:**
    - The accident data contains the lat-long coordinates of every accident which has been mapped to the nearest street in the data generated from OpenStreetMaps. 
    - The road network has the lat-long coordinates of every intersection/node in the graph. 
    - We assume that the accident takes place at some point between two nodes. 
    - Let’s assume the accident takes place at point C which is on the street between nodes A and B. 
    - The distance AC + BC should ideally be equal to AB assuming the street AB is a straight road. 
    - Using the above methodology, we iterate over all the streets and map the accident to the street where the difference of distances (AB - (AC+BC)) is the lowest.

- **Traffic Data, Road Network:**
    - Traffic Data contains the names of the streets defined by Department of Transportation
    - Road Network has the names of streets given by OpenStreetMaps
    - The names have to be manually mapped to combine both these data sources


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
