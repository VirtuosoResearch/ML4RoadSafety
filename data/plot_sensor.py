import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pandas as pd

request = cimgt.OSM()

df = pd.read_csv('sensor_graph/graph_sensor_locations.csv')
# df = pd.read_csv('sensor_graph/graph_sensor_locations_bay.csv')
print(df.head())
lons = df['longitude'].values
lats = df['latitude'].values

print(min(lons), max(lons), min(lats), max(lats))

fig, ax = plt.subplots(figsize=(10,16),
                       subplot_kw=dict(projection=request.crs))
# ax.set_extent([-117.5, -119, 33.5, 34.5]) # la
ax.set_extent([-122.5, -121.5, 37, 38]) # bay area

ax.add_image(request, 8)
# ax.states(resolution='50m')
data_crs = ccrs.PlateCarree()
ax.plot(lons,lats, color='black', marker='.', ms=3, mew=.5, transform=data_crs, linestyle='None')


plt.savefig('bay.png')