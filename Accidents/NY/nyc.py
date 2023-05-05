import pandas as pd


'''
Convert linestring to list of coordinates
'''
def linestr2coors(linestr):
    # locate '(' and ')'
    index = [i for i, x in enumerate(linestr) if x == '(' or x == ')']
    tmp_str = linestr[index[0] + 1:index[1]]
    tmp_str_list = tmp_str.split(',')
    coordi_list = []
    print(tmp_str_list)
    for i in range(0, len(tmp_str_list)):
        lon_lat = tmp_str_list[i].split(' ')
        coordi_list.append((float(lon_lat[-2].strip()), float(lon_lat[-1])))
    return coordi_list




df = pd.read_csv('CSCL_PUB_Centerline.csv')

# print(df['the_geom'].head())
# print(df['the_geom'][0])
print(linestr2coors(df['the_geom'][0]))


# TODO: save a list of coordinates to a file
# TODO: convert the coordinates to a graph (adjacent list)
# TODO: Map intersections to roads and find shortest distance between two intersections (possible)
# TODO: find out which measure error is acceptable