import numpy as np


def days_with_collision(loc, k, collision_thres):
    if loc == 'la':
        x = np.load('METR-LA/node_values.npy')
        y = np.load('METR-LA/accident_data.npy')
    elif loc == 'bay':
        x = np.load('PEMS-BAY/pems_node_values.npy')
        y = np.load('PEMS-BAY/accident_data.npy')
    else:
        return
    
    days = []
    for i in range(y.shape[0]):
        if np.sum(y[i]) > collision_thres:
            days.append(i)
    
    days = np.array(days)

    return days



# print(days_with_collision('la', 1, 10))
