import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

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
        

# print(intersections)
g_index = np.arange(0, 112, 1).reshape(16, 7)
for i in range(g_index.shape[0]):
    for j in range(g_index.shape[1]-1):
        G.add_edge(g_index[i][j], g_index[i][j+1])
        
g_index = g_index.T
for i in range(g_index.shape[0]):
    for j in range(g_index.shape[1]-1):
        G.add_edge(g_index[i][j], g_index[i][j+1])

print(node_index)
# print(G.nodes.data())
print(G.number_of_nodes())
print(G.number_of_edges())

# TODO: select a small range of time to upload the crash information

'''
Given a node and a list of nodes, find the nearest node
'''
def nearest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return nodes[np.argmin(dist_2)]

'''
Find the nearest distance between a node and a list of nodes
'''
def nearest_dist(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.min(dist_2)


'''
Calulate the distance between two nodes (latitude and longitude)
'''
def dist(node1, node2):
    return np.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)


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
        
print(len(locations)) 
print(list(G.edges()))
