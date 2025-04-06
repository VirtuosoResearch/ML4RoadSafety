import os
import math
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import requests
import time

ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0

def lonLatToMeters(lon, lat):
    mx = lon * ORIGIN_SHIFT / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * ORIGIN_SHIFT / 180.0
    return mx, my

def metersToLonLat(mx, my):
    lon = (mx / ORIGIN_SHIFT) * 180.0
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lon, lat

def DownloadMapbox(min_lat, min_lon, max_lat, max_lon, zoom, outputname):
    MAPBOX_API_KEY = "pk.eyJ1Ijoia2F0aHl4NTMiLCJhIjoiY2x0N3VvdXYwMG93dTJpdDNyNnplZXpsMSJ9.d6pbjsi8aFhf6PllF3Wp7g"
    bbox = f"[{min_lon},{min_lat},{max_lon},{max_lat}]"
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{bbox}/1280x1280?access_token={MAPBOX_API_KEY}"

    retry_timeout = 10
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            with open(outputname, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Download failed, retrying in {retry_timeout}s")
            time.sleep(retry_timeout)
            retry_timeout = min(retry_timeout + 10, 60)

def download_map_tiles(min_lat, min_lon, max_lat, max_lon, folder="tiles/", start_lat=40.7128, start_lon=-74.0060, resolution=1024, padding=128, zoom=19):
    resolution_lat = 1.0 / 111111.0
    resolution_lon = 1.0 / (111111.0 * math.cos(start_lat / 360.0 * (math.pi * 2)))

    x, y = lonLatToMeters(start_lon, start_lat)
    w = 2 * math.pi * 6378137 / math.pow(2, zoom)

    lon2, lat2 = metersToLonLat(x + w, y + w)
    lon1, lat1 = metersToLonLat(x - w, y - w)

    angle_per_image_lat = lat2 - lat1
    angle_per_image_lon = lon2 - lon1

    start_lat -= angle_per_image_lat * 0.5
    start_lon -= angle_per_image_lon * 0.5

    ilat_min = int(math.floor((min_lat - start_lat) / angle_per_image_lat))
    ilon_min = int(math.floor((min_lon - start_lon) / angle_per_image_lon))
    ilat_max = int(math.floor((max_lat - start_lat) / angle_per_image_lat))
    ilon_max = int(math.floor((max_lon - start_lon) / angle_per_image_lon))

    lat_n = ilat_max - ilat_min + 1
    lon_n = ilon_max - ilon_min + 1

    tile_info = []
    for i in range(ilat_max, ilat_min - 1, -1):
        for j in range(ilon_min, ilon_max + 1):
            if (j - ilon_min)!=1 or (ilat_max - i)!=1: continue
            
            filename = folder + f"sat_{j - ilon_min}_{ilat_max - i}.png"
            tile_min_lat = start_lat + i * angle_per_image_lat
            tile_max_lat = start_lat + (i + 1) * angle_per_image_lat
            tile_min_lon = start_lon + j * angle_per_image_lon
            tile_max_lon = start_lon + (j + 1) * angle_per_image_lon

            # if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f"Downloading tile at lat {(tile_min_lat + tile_max_lat) / 2}, lon {(tile_min_lon + tile_max_lon) / 2}")
            DownloadMapbox(tile_min_lat, tile_min_lon, tile_max_lat, tile_max_lon, zoom, filename)

            tile_info.append({
                'filename': filename,
                'row': ilat_max - i,
                'col': j - ilon_min
            })

    return tile_info, lat_n, lon_n, angle_per_image_lat, angle_per_image_lon, start_lat, start_lon

def get_satellite_patch_with_marker(
    lat,
    lon,
    meters=100,
    image_size=512,
    zoom=19,
    folder="./tiles/",
    output_path=None,
    marker_color=(255, 0, 0),
    marker_radius=5
):
    delta_lat = meters / 111111.0
    delta_lon = meters / (111111.0 * math.cos(math.radians(lat)))
    min_lat = lat - delta_lat / 2
    max_lat = lat + delta_lat / 2
    min_lon = lon - delta_lon / 2
    max_lon = lon + delta_lon / 2

    tile_info, lat_n, lon_n, angle_lat, angle_lon, start_lat, start_lon = download_map_tiles(
        min_lat, min_lon, max_lat, max_lon,
        folder=folder,
        start_lat=lat,
        start_lon=lon,
        resolution=1024,
        padding=128,
        zoom=zoom
    )
    if output_path:
        mark_center_on_image("./tiles/sat_1_1.png", output_path=output_path)

def mark_center_on_image(image_path, output_path=None, marker_color=(255, 0, 0), marker_radius=5):
    if not os.path.isfile(image_path):
        print(f"image doesn't exist: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    center = (w // 2, h // 2)

    draw.ellipse([
        center[0] - marker_radius,
        center[1] - marker_radius,
        center[0] + marker_radius,
        center[1] + marker_radius
    ], fill=marker_color)

    if output_path is None:
        output_path = image_path

    img.save(output_path)
    print(f"marks done : {output_path}")

if __name__ == "__main__":
    lat = 42.8767484
    lon = -71.005654

    get_satellite_patch_with_marker(
        lat=lat,
        lon=lon,
        meters=200,
        image_size=512,
        output_path="./tiles/Marked_the_Center.png"
    )
    