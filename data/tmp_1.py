from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset()
print(dataset.features[0].shape)
print(dataset.targets[0].shape)
# print(dataset.targets)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h.log_softmax(dim=-1)
    

    

def get_y():
    import numpy as np
    import pandas as pd
    import sqlite3
    import argparse
    from datetime import datetime, timedelta
    import datetime
    from math import ceil

    def compute_distance_two_points(lon1, lat1, lon2, lat2):
        '''
        Calculate the great circle distance between two points, unit is km
        '''
        # approximate radius of earth in km
        R = 6373.0

        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c

        return distance


    def on_highway(lon, lat, lons, lats, k):
        for i in range(len(lons)):
            if compute_distance_two_points(lon, lat, lons[i], lats[i]) <= k:
                return True
        return False


    def compute_min_radius(lons, lats):
        '''
        compute the minimum distance between two points in a set of points
        '''
        min_distance = np.inf
        for i in range(len(lons)):
            for j in range(i+1, len(lons)):
                distance = compute_distance_two_points(lons[i], lats[i], lons[j], lats[j])
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance/2
        
    def round_up_to_neareast_5_mins(m):
        '''
        m is a datetime object and we round up to the nearest 5 mins
        '''
        if m.minute % 5 == 0 and m.second == 0:
            return m
        m -= timedelta(minutes=m.minute % 5, seconds=m.second)
        m += timedelta(minutes=5)
        return m
    
    def str2datetime(d, t):
        tmp = str(d) + ' ' + str(t)
        d = datetime.datetime.strptime(tmp, '%Y-%m-%d %H:%M:%S')
        return d
            

    with sqlite3.connect("switrs.sqlite") as con:

        query = (
            "SELECT collision_date, collision_time, latitude, longitude "
            "FROM collisions "
            "WHERE collision_date IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL"
        )

        # Construct a Dataframe from the results
        df = pd.read_sql_query(query, con)

    # 193176 data points

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.name = 'bay'
    args.k = 1.0

    # -----------------------------------------------------------------------

    if args.name == 'la':
        dataset = pd.read_csv('sensor_graph/graph_sensor_locations.csv')
        start = datetime.datetime(2012, 3, 1)
        end = datetime.datetime(2012, 6, 27)
    elif args.name == 'bay':
        dataset = pd.read_csv('sensor_graph/graph_sensor_locations_bay.csv')
        start = datetime.datetime(2017, 1, 1)
        end = datetime.datetime(2017, 6, 30)

    df_parsed = df[(df['collision_date'] >= str(start)) & (df['collision_date'] <= str(end))]
    df_parsed.reset_index(drop=True, inplace=True)
    a = df_parsed['collision_time'][0]


    lons = dataset['longitude'].values
    lats = dataset['latitude'].values

    # radius = compute_min_radius(lons, lats)
    # print('radius', radius)
    col_lon = []
    col_lat = []

    for i in range(len(df_parsed)):
        lon = df_parsed['longitude'][i]
        lat = df_parsed['latitude'][i]
        if on_highway(lon, lat, lons, lats, args.k):
            col_lon.append(lon)
            col_lat.append(lat)
        
    # -----------------------------------------------------------------------
    print('number of collisions', len(col_lon))


    node_values = np.load('PEMS-BAY/pems_node_values.npy')

    # First col: speed
    # Second col: time of the day (5 minutes interval)
    # Third col: if there is a collision
    # Fourth col: TBD


    # -----------------------------------------------------------------------
    # here collision comes in

    # collision corrsponding to the sensors
    # add collision to node_values

    def accident_to_sensor(lon, lat, lons, lats, k):
        locs = []
        for i in range(len(lons)):
            if compute_distance_two_points(lon, lat, lons[i], lats[i]) <= k:
                locs.append(i)
        return locs

    def accident_time_idx(time, start):
        '''
        time is a datetime object
        '''
        return ceil((time - start).total_seconds() / 300)

    # dict structure: key is the sensor id, value is a list of collision time indices

    accid_data = np.zeros((node_values.shape[0], node_values.shape[1], 1))

    for i in range(len(df_parsed)):
        accid_sensors = accident_to_sensor(df_parsed['longitude'][i], df_parsed['latitude'][i], lons, lats, args.k)
        if len(accid_sensors) == 0:
            continue
        
        try:
            accid_time = str2datetime(df_parsed['collision_date'][i], df_parsed['collision_time'][i])
        except:
            continue
        time_idx = accident_time_idx(accid_time, start)
        
        for i in accid_sensors:
            try :
                accid_data[time_idx, i, 0] = 1
            except:
                print(time_idx, i)
                continue

    # print(accid_data)

    aggregate_data = np.concatenate((node_values, accid_data), axis=2)

    return accid_data


    # -----------------------------------------------------------------------
    # add some graph features :)



def adj2edge_index(adj):
    edge_index = []
    edge_weight = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                edge_index.append([i,j])
                edge_weight.append(adj[i][j])
    return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_weight)

import numpy as np

t = np.load('PEMS-BAY/pems_node_values.npy')
x = []
for i in range(1000):
    x.append(t[i])

y_0 = get_y()
y_0 = y_0.reshape(y_0.shape[0], y_0.shape[1])
y = []
for i in range(1000):
    y.append(y_0[i])

adj = 'PEMS-BAY/pems_adj_mat.npy'

def dataset2graph_signal(x, y, adj):
    adj_mx = np.load(adj)
    edge_index, edge_weight = adj2edge_index(adj_mx)
    
    features = x
    targets = y
    
    return StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
    
my_dataset = dataset2graph_signal(x, y, adj)
print(my_dataset.features[0].shape)
print(my_dataset.targets[0].shape)
# my_dataset = StaticGraphTemporalSignal(t, adj2edge_index(np.load('METR-LA/adj_mx.npy'))[0], adj2edge_index(np.load('METR-LA/adj_mx.npy'))[1])

train_dataset, test_dataset = temporal_signal_split(my_dataset, train_ratio=0.8)

from tqdm import tqdm

model = RecurrentGCN(node_features = 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()


# TODO: add a classfication critiria to the model and print out accuracy

model.eval()
cost = 0
correct = 0
total = 0
true_postive = 0
false = 0
y_total = 0
error = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)

    for i in range(len(y_hat)):
        total += 1
        y_total += snapshot.y[i]
        error += abs(y_hat[i, 0]-snapshot.y[i])
        if abs(y_hat[i, 0]-snapshot.y[i]) < 0.5:
            correct += 1
            if y_hat[i, 0] > 0.5:
                true_postive += 1
        else:
            false += 1/2
print('error: {}'.format(error/total))
print('y_total: {}'.format(y_total))   
print('F-score: {}'.format(true_postive/(true_postive+false)))
print('Number of correct predictions: {}'.format(correct))
print('Number of total predictions: {}'.format(total))
print('Accuracy: {:.4f}'.format(correct/total))
# print("MSE: {:.4f}".format(cost))
