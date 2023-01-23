import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pandas as pd
import sqlite3
import argparse
from datetime import datetime, timedelta, date

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
parser.add_argument('--k', type=float) # k means max distance between collision and sensor

args = parser.parse_args()

# -----------------------------------------------------------------------
request = cimgt.OSM()
fig, ax = plt.subplots(figsize=(10,16),
                       subplot_kw=dict(projection=request.crs))

if args.name == 'la':
    ax.set_extent([-117.5, -119, 33.5, 34.5]) # la
    dataset = pd.read_csv('sensor_graph/graph_sensor_locations.csv')
    start = datetime(2012, 3, 1)
    end = datetime(2012, 6, 27)
elif args.name == 'bay':
    ax.set_extent([-122.5, -121.5, 37, 38])
    dataset = pd.read_csv('sensor_graph/graph_sensor_locations_bay.csv')
    start = datetime(2017, 1, 1)
    end = datetime(2017, 5, 31)

df_parsed = df[(df['collision_date'] > str(start)) & (df['collision_date'] < str(end))]
df_parsed.reset_index(drop=True, inplace=True)
print(df_parsed.head())

lons = dataset['longitude'].values
lats = dataset['latitude'].values

radius = compute_min_radius(lons, lats)
print('radius', radius)
col_lon = []
col_lat = []

for i in range(len(df_parsed)):
    lon = df_parsed['longitude'][i]
    lat = df_parsed['latitude'][i]
    if on_highway(lon, lat, lons, lats, args.k):
        col_lon.append(lon)
        col_lat.append(lat)
    
# -----------------------------------------------------------------------

print(len(col_lon))


# ax.set_extent([-126, -113, 32, 43]) # la
# ax.set_extent([-122.5, -121.5, 37, 38]) # bay area

ax.add_image(request, 8)
# ax.states(resolution='50m')
data_crs = ccrs.PlateCarree()
ax.plot(col_lon, col_lat, color='black', marker='.', ms=3, mew=.5, transform=data_crs, linestyle='None')

plt.savefig('collision_on_'+args.name+'_highway.png')
