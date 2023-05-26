import numpy as np

def organize_edges(edges):
    mask = edges[:, 0] > edges[:, 1]
    edges[mask, 0], edges[mask, 1] = edges[mask, 1], edges[mask, 0]
    return edges