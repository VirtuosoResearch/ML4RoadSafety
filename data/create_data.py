import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import datetime
from math import ceil
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def create_dataset(name, k=1.0):

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

    # -----------------------------------------------------------------------

    if name == 'la':
        sensor_loc = pd.read_csv('sensor_graph/graph_sensor_locations.csv')
        start = datetime.datetime(2012, 3, 1)
        end = datetime.datetime(2012, 6, 27)
    elif name == 'bay':
        sensor_loc = pd.read_csv('sensor_graph/graph_sensor_locations_bay.csv')
        start = datetime.datetime(2017, 1, 1)
        end = datetime.datetime(2017, 6, 30)

    df_parsed = df[(df['collision_date'] >= str(start)) & (df['collision_date'] <= str(end))]
    df_parsed.reset_index(drop=True, inplace=True)


    lons = sensor_loc['longitude'].values
    lats = sensor_loc['latitude'].values

    # radius = compute_min_radius(lons, lats)
    # print('radius', radius)
    col_lon = []
    col_lat = []

    for i in range(len(df_parsed)):
        lon = df_parsed['longitude'][i]
        lat = df_parsed['latitude'][i]
        if on_highway(lon, lat, lons, lats, k):
            col_lon.append(lon)
            col_lat.append(lat)
        
    # -----------------------------------------------------------------------
    print('number of collisions', len(col_lon))

    if name == 'la':
        node_values = np.load('METR-LA/node_values.npy')
    elif name == 'bay':
        node_values = np.load('PEMS-BAY/pems_node_values.npy')

    # First col: speed
    # Second col: time of the day (5 minutes interval)
    # Third col: if there is a collision

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
        accid_sensors = accident_to_sensor(df_parsed['longitude'][i], df_parsed['latitude'][i], lons, lats, k)
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

    # aggregate_data = np.concatenate((node_values, accid_data), axis=2)

    # x = []
    # for i in range(1000):
    #     x.append(node_values[i])
    
    # TODO: change the y to the accident data, y is to be produced
    # y_0 = get_y()
    # y_0 = y_0.reshape(y_0.shape[0], y_0.shape[1])
    # y = []
    # for i in range(1000):
    #     y.append(y_0[i])
        
    if name == 'bay':
        adj = 'PEMS-BAY/pems_adj_mat.npy'
    elif name == 'la':
        adj = 'METR-LA/adj_mat.npy'

    def dataset2graph_signal(x, y, adj):
        adj_mx = np.load(adj)
        edge_index, edge_weight = adj2edge_index(adj_mx)
        
        features = x
        targets = y
        
        return StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)
        
    # my_dataset = dataset2graph_signal(x, y, adj)

    return accid_data


    # -----------------------------------------------------------------------
    # add some graph features 

dataset = create_dataset('bay', 1)
# np.save('METR-LA/accident_data.npy', dataset)
np.save('PEMS-BAY/accident_data.npy', dataset)
print(dataset.shape)
print(np.sum(dataset))