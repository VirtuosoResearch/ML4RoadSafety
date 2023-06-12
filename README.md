# Code repository for Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis

This repository includes the code for reproducing our results, including the code for collecting our dataset, and the implemention of GNNs for predicting the accident labels. Our dataset consists of 9 million accident records gathered from 8 states across the United States. The link to download our processed data files is https://dataverse.harvard.edu/privateurl.xhtml?token=add1d658-0e71-4007-9735-7976efb8de5e.

### Features

- Diverse and extensive collection of traffic accident datasets
- Easy-to-use dataset loader for convenient data loading, fully compatible with the graph deep learning framework, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- Example code scripts for training and evaluating graph neural networks on our datasets
- Detailed documentation on the data collection process using publicly available sources

### Installation

To use the traffic accidents dataset and its associated tools, follow these steps:

1. Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/anonymous-researchcode/ML4RoadSafety.git
```

2. Install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

**Requirments**: we list the mostly used packages in this repository as follows.

- Python>=3.6
- PyTorch>=1.10
- torch-geometric>=2.0.3
- numpy>=1.19.0
- torch-geometric-temporal>=0.54.0

### Quick Start: Package Usage

In the following instructions, we provide a quick overview of how to use our package for analyzing traffic accidents on road networks. We first describe the usage of our easy-to-use data loader and then provide a script to train and evaluate graph neural networks on our datasets.

**Data Loader**

We prepare a data loader named `TrafficAccidentDataset` that loads the road network as the data format of Pytorch Geometric. Then, we can use a few lines of code with `load_monthly_data(year, month)` to load the accident labels and network features for a particular month. To run these commands, first, download the zip file for that state from the [data link](https://dataverse.harvard.edu/privateurl.xhtml?token=add1d658-0e71-4007-9735-7976efb8de5e), unzip this file.

```python
from ml_for_road_safety import TrafficAccidentDataset

# Creating the dataset as PyTorch Geometric dataset object
dataset = TrafficAccidentDataset(state_name = "MA")

# Loading the accident records and traffic network features of a particular month
data = dataset.load_monthly_data(year = 2022, month = 1)

# Pytorch Tensors storing the list of edges with accidents and accident numbers
accidents, accident_counts = data["accidents"], data["accident_counts"]

# Pytorch Tensors of node features, edge list, and edge features
x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]
```

With these functions, you can split the datasets into different months for training and evaluation. Additionally, we provide an `Evaluator` module that can be easily integrated into your training or test loop:

```python
from ml_for_road_safety import Evaluator

# Get an evaluator for accident prediction, e.g., the regression task. 
evaluator = Evaluator(type = "regression")

# Iterating over months in a period of time for training
for month in train_months:
    ...
    pred_accident_counts = model(x, edge_index, edge_attr)
    # Compute the prediction loss
    loss = evaluator.criterion(pred_accident_counts, true_accident_counts)
    
# Iterating over months in a period of time for testing
for month in test_months:
    ...
    # Compute the prediction performance
    pred_accident_counts = model(x, edge_index, edge_attr)
    results_dict = evaluator.eval(pred_accident_counts, true_accident_counts)
    print(results_dict['MAE'])
```

**Training and Evaluating GNNs**

To simplify the process of training and evaluating graph neural networks (GNNs) on our dataset for a specific state, we provide a Python script called `train.py`. You can customize the training process by specifying the following parameters:

- `--state_name` specifies the state of the dataset. Choose among `DE`, `IA`, `IL`, `MA`, `MD`, `MN`, `MT`, `NV`. 

- `--node_feature_type` specifies the type of node features. Choose among `centrality`, `node2vec`, and `deepwalk`. See `README.md` under the `embeddings` folder for the instructions to generate node features.
- `--encoder` specifies the model to encode node features. `none` indicate only using MLP as the predictor. We provide the following encoders: `gcn`, `graphsage`, `gin`, and `dcrnn`. 
  -  `--num_gnn_layers` specifies the number of encoder layers.
  - `--hidden_channels` specifies the hidden model width. 
- `--train_years`, `--valid_years`, and `--test_years` specifies the splitting of datasets for training, validation, and testing. 

Here is an example of bash script for training a GCN model on MA dataset below:

```bash
python train.py --state_name MA --node_feature_type node2vec\
    --encoder gcn --num_gnn_layers 2 \
    --epochs 100 --lr 0.001 --runs 3 \
    --load_dynamic_node_features\
    --load_static_edge_features\
    --load_dynamic_edge_features\
    --train_years 2002 2003 2004 2005 2006 2007 2008 \
    --valid_years 2009 2010 2011 2012 2013 2014 2015 \
    --test_years  2016 2017 2018 2019 2020 2021 2022 \
    --device 0
```
