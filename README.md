# Overview

This repository includes the code for reproducing our results in "Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis," which will be presented at NeurIPS'23, New Orleans.

We document the code for collecting our dataset and the implementation of GNNs for predicting the accident labels, including
- Easy-to-use dataset loader for convenient data loading, fully compatible with the graph deep learning framework, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).
- Example code scripts for training and evaluating graph neural networks on our datasets.
- Detailed documentation on the data collection process using publicly available sources, resulting in 9 million accident records gathered from 8 states across the US. [Link](https://dataverse.harvard.edu/privateurl.xhtml?token=add1d658-0e71-4007-9735-7976efb8de5e) to download our processed data files.

### Quick Start

We provide a quick overview of how to use our package for analyzing traffic accidents on road networks. We describe the usage of our easy-to-use data loader, named `TrafficAccidentDataset` that loads the road network as the data format of Pytorch Geometric. Then, we can use a few lines of code with `load_monthly_data(year, month)` to load the accident labels and network features for a particular month. Our code automatically downloads datasets into the `/ml_for_road_safety/data/` folder. To run these commands, one can also manually download the zip file for that state from the [data link](https://doi.org/10.7910/DVN/V71K5R), unzip this file under the `/ml_for_road_safety/data/` folder.

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

which includes the most used packages such as

- Python>=3.6
- PyTorch>=1.10
- torch-geometric>=2.0.3
- numpy>=1.19.0
- torch-geometric-temporal>=0.54.0
- pyDataverse
