python train.py --state_name MA --node_feature_type deepwalk\
    --encoder gcn --num_gnn_layers 2 \
    --epochs 30 --lr 0.001 --runs 3 \
    --load_dynamic_node_features\
    --load_static_edge_features\
    --load_dynamic_edge_features\
    --train_years 2008 2009\
    --valid_years 2010 2011 \
    --test_years  2012\
    --device 0

# python train.py --state_name MA --node_feature_type centrality\
#     --encoder gcn --num_gnn_layers 2 \
#     --epochs 30 --lr 0.001 --runs 3 \
#     --load_dynamic_node_features\
#     --load_static_edge_features\
#     --load_dynamic_edge_features\
#     --train_years 2009 2010 2011 2012 2013 2014 2015 2016 \
#     --valid_years 2017 2018 2019 2020 2021 \
#     --test_years  2022 2023 2024\
#     --device 0