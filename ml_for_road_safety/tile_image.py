import math
import os
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import pandas as pd
from tqdm import tqdm

MAPBOX_ACCESS_TOKEN = "pk.eyJ1IjoiemluaXV6aGFuZyIsImEiOiJjbTFzZDhzNXkwNWdvMmtwcnprajdnYnRjIn0.1_B5eBhh0s3Pw0UlSlThyQ"
TILE_SIZE = 512 

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def deg2pixel(lat_deg, lon_deg, zoom, tile_size=512):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n * tile_size
    y = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * tile_size
    return x, y

def tile_url(x, y, z, access_token):
    return f"https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token={access_token}"

def download_tile(x, y, z, access_token):
    url = tile_url(x, y, z, access_token)
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise Exception(f"Tile download failed: {url}, status {response.status_code}")

def draw_marker(image, marker_color=(255, 0, 0), radius=6):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    center = (w // 2, h // 2)
    draw.ellipse([
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius
    ], fill=marker_color)
    return image

def download_tile_image_like_static_image(
    center_lat,
    center_lon,
    zoom,
    output_path,
    output_size=(1280, 1280),
    access_token=MAPBOX_ACCESS_TOKEN,
    tile_size=TILE_SIZE,
    draw_center_marker=True
):

    num_tiles_x = math.ceil(output_size[0] / tile_size) + 2
    num_tiles_y = math.ceil(output_size[1] / tile_size) + 2


    center_x_tile, center_y_tile = deg2num(center_lat, center_lon, zoom)

    x_start = center_x_tile - num_tiles_x // 2
    y_start = center_y_tile - num_tiles_y // 2

    # print("num: ", num_tiles_x*num_tiles_y)
    stitched_image = Image.new("RGB", (num_tiles_x * tile_size, num_tiles_y * tile_size))

    for dx in range(num_tiles_x):
        for dy in range(num_tiles_y):
            x = x_start + dx
            y = y_start + dy
            try:
                tile = download_tile(x, y, zoom, access_token)
                stitched_image.paste(tile, (dx * tile_size, dy * tile_size))
            except Exception as e:
                print(f"Tile ({x},{y}) download failed: {e}")

    x_origin = x_start * tile_size
    y_origin = y_start * tile_size

    center_global_x, center_global_y = deg2pixel(center_lat, center_lon, zoom, tile_size)

    center_x = int(center_global_x - x_origin)
    center_y = int(center_global_y - y_origin)

    left = center_x - output_size[0] // 2
    upper = center_y - output_size[1] // 2
    right = center_x + output_size[0] // 2
    lower = center_y + output_size[1] // 2

    final_img = stitched_image.crop((left, upper, right, lower))

    if draw_center_marker:
        final_img = draw_marker(final_img)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img.save(output_path)
    # print(f"Saved: {output_path}")

if __name__ == "__main__":
    lat, lon = 42.34540, -71.08284
    path = "/home/michael/project/data/MLRoadSafety/Road_Networks/NV/Road_Network_Nodes_NV.csv"
    nodes = pd.read_csv(path)
    cnt =0
    for idx, node in tqdm(nodes.iterrows(), total=len(nodes)):
        id, lat, lon = int(node["node_id"]), node["y"], node["x"]
        if cnt>=29950: break
        if not os.path.exists(f"/home/michael/project/data/Nodes_NV/{id}.png"):
            download_tile_image_like_static_image(
                center_lat=lat,
                center_lon=lon,
                zoom=19,
                output_path=f"/home/michael/project/data/Nodes_NV/{id}.png",
                output_size=(1280, 1280),
                draw_center_marker=True
            )
