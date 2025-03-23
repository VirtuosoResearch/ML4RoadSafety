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