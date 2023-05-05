import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx


'''
Calculate the distance between two points with latitude and longitude
'''
def distance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance


'''
Calculate the nearest node to a given point, with index and distance
'''
def nearest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return nodes[np.argmin(dist_2)]


'''
Nearest distance with the node (0.1)
'''
def nearest_distance(node, nodes):
    nodes = np.asarray(nodes)
    nodes = np.reshape(nodes, (-1, 2))
    distances = []
    for i in range(len(nodes)):
        distances.append(distance(node[0], node[1], nodes[i][0], nodes[i][1]))
    return min(distances)


'''
Test if a node is within the area
'''
def in_area(node, area):
    area = np.asarray(area)
    area = np.reshape(area, (-1, 2))
    for i in range(len(area)):
        if distance(node[0], node[1], area[i][0], area[i][1]) <= 0.1:
            return True
    return False


'''
Generate edge_index from a nx.Graph
'''
def generate_edge_index(G):
    list_edges = G.edges()
    list_edges = np.asarray(list_edges)
    
    edge_index = list_edges.T
    return edge_index


df = pd.read_csv('../nyc_crash.csv')
start = datetime(2017, 1, 1, 0, 0)
end = datetime(2017, 1, 31, 23, 59)
locations = []

for i in range(len(df)):
    tmp = datetime(int(df['year'][i]), int(df['month'][i]), int(df['day'][i]), int(df['hour'][i]), int(df['minute'][i]))
    if tmp >= start and tmp <= end:
        if np.isnan(df['lat'][i]):
            continue
        locations.append((df['lat'][i], df['lon'][i]))
        

print('Number of crashes: ', len(locations))


df = pd.read_csv('intersection.csv')


# index 16 to split

end = len(df)

intercept = [(0,0)]

for i in range(17, end):
    tmp = (df['latitude'][i] - df['latitude'][0], df['longitude'][i] - df['longitude'][0])
    intercept.append(tmp)

print(len(intercept))

G = nx.Graph()
node_index = 0
intersections = np.zeros((16, 7, 2))

for i in range(16):
    for j in range(7):
        intersections[i][j][0] = df['latitude'][i]+intercept[j][0]
        intersections[i][j][1] = df['longitude'][i]+intercept[j][1]
        G.add_node(node_index, pos=(intersections[i][j][0], intersections[i][j][1]))
        node_index += 1

valid_locations = 0
for num, i in enumerate(locations):
    ''' In the area, area should be the small network constructed'''
    if in_area(i, intersections):
        valid_locations += 1
        
print('Number of crashes in the area: ', valid_locations) # 240 for ~30 days

