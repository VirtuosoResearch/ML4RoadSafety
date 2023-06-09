import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pandas as pd

df = pd.read_csv('Traffic_Collision_Data_from_2010_to_Present.csv') # Los Angeles traffic collision data from 2010 to present

# 193176 data points

request = cimgt.OSM()

lats = []
lons = []

# remove nan values
df['Location'] = df['Location'].dropna()
for i in range(len(df)):
    if df['Location'][i] == 'nan':
        continue
    try:
        lat, lon = eval(df['Location'][i])
    except:
        continue
    if lat == 0.0 or lon == 0.0:
        continue
    lats.append(lat)
    lons.append(lon)

print(min(lons), max(lons), min(lats), max(lats))

fig, ax = plt.subplots(figsize=(10,16),
                       subplot_kw=dict(projection=request.crs))
ax.set_extent([-117.5, -119, 33.5, 34.5]) # la
# ax.set_extent([-122.5, -121.5, 37, 38]) # bay area

ax.add_image(request, 8)
# ax.states(resolution='50m')
data_crs = ccrs.PlateCarree()
ax.plot(lons,lats, color='black', marker='.', ms=3, mew=.5, transform=data_crs, linestyle='None')

plt.savefig('collision.png')

