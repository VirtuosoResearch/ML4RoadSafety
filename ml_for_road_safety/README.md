# Training and Evaluating GNNs

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
