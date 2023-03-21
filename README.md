# Predicting road accidents using graph neural networks

This is a machine learning project that uses Graph Neural Networks (GNNs) to predict road accidents. We have manually collected several datasets. The data is from the Massachusetts Department of Transportation (MassDOT) and the New York City Police Department. The goal is to predict the likelihood of an accident at a given location in a given time interval. 

## Introduction

Road accidents are a major cause of injury and death worldwide. Machine learning techniques can help us to predict the likelihood of accidents and prevent them from happening. 

## Datasets

We have collected several datasets that we use in this project:

- **Massachusetts:** A dataset that contains information about road accidents in Massachusetts from January 2015 to February 2023. The dataset contains 3000+ accidents.

- **New York City:** A dataset that contains information about road accidents in New York City from April 2014 to March 2023. The dataset contains 50,000+ accidents.

[Download link](https://drive.google.com/drive/folders/1PHIkgoKkugj6rMvkbjxpJbmzVn69dm8e)

## Baseline Models

We ran experiments on several baseline models, including:

- **(DCRNN)[https://arxiv.org/abs/1707.01926]:** A graph neural network that consists of a diffusion convolutional layer to capture spatial dependencies, a recurrent layer to capture temporal dependencies, and a pooling layer to aggregate information across multiple time steps.

- **Graph-WaveNet:** A graph convolutional neural network that utlizes a self-adaptive adjacency matrix to capture spatial-temporal dependencies simultaneously.

- **STGCN:** A graph convolutional neural network that uses a spatio-temporal graph convolutional network architecture to aggregate information across multiple time steps in traffic forecasting tasks.

## Installation

To install the required packages, run the following command:
    
    pip install -r requirements.txt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.