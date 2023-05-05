import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import pandas as pd
import sqlite3

with sqlite3.connect("switrs.sqlite") as con:

    query = (
        "SELECT collision_date, collision_time, latitude, longitude "
        "FROM collisions "
        "WHERE collision_date IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL"
    )

    # Construct a Dataframe from the results
    df = pd.read_sql_query(query, con)

# 193176 data points

request = cimgt.OSM()


fig, ax = plt.subplots(figsize=(10,16),
                       subplot_kw=dict(projection=request.crs))
ax.set_extent([-126, -113, 32, 43]) # la
# ax.set_extent([-122.5, -121.5, 37, 38]) # bay area

ax.add_image(request, 8)
# ax.states(resolution='50m')
data_crs = ccrs.PlateCarree()
ax.plot(df['longitude'], df['latitude'], color='black', marker='.', ms=3, mew=.5, transform=data_crs, linestyle='None')

plt.savefig('highway.png')

