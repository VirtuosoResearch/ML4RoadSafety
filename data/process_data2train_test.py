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
    if str(d) == str(start) or str(d) == str(end):
        print(d)
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
parser.add_argument('--name', type=str, default='la')
parser.add_argument('--k', type=float, default=1.0) # k means max distance between collision and sensor

args = parser.parse_args()

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


node_values = np.load('METR-LA/node_values.npy')

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
        accid_data[time_idx, i, 0] = 1

# print(accid_data)

aggregate_data = np.concatenate((node_values, accid_data), axis=2)
print(aggregate_data.shape)
print(np.sum(aggregate_data[:, :, 2]))
print(aggregate_data[0].shape)
print(np.sum(aggregate_data[0, :, 2]))

print(accid_data.shape)


# -----------------------------------------------------------------------
# add some graph features :)

