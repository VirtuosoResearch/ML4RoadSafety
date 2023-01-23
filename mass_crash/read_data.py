import pandas as pd

# Read the data mass_fatality.xlsx
df = pd.read_excel('mass_fatality.xlsx')
# print(df["Unnamed: 11"])
# print(df["Unnamed: 12"])


# starting 42
# 11: one every two rows
# 12: every row

new_df = None

col_type = []
locs = []
dates = []

for i in range(42, 5979,2):
    col_type.append(df["Unnamed: 11"][i])
    dates.append(df["Unnamed: 12"][i])
    locs.append(df["Unnamed: 12"][i+1])
    
    # print(i, end='')
# print(dates)
print(dates[0].split(' '))

import csv

# writer = csv.writer(open('mass_fatality.csv', 'w'))

def col_type2info(col_type):
    gender = col_type[-1]
    if gender != 'M' and gender != 'F':
        return KeyError
    
    reason = col_type.split(',')[0]
    
    pass
    
    

with open('mass_fatality.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    
    writer.writerow(['year', 'month', 'day', 'hour', 'minute', 'AM/PM', 'location', 'type', 'gender'])
    
    for i in range(len(col_type)):
        day, time, am_pm = dates[i].split(' ')
        month, day, year = day.split('/')
        hour, minute = time.split(':')
        year = int(year[:-1])
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        
        type_1, gender = col_type[i].split(', ')
        
        # "OPERATOR, M"
        # "MOTORCYCLE: OPERATOR, M"
        # "OPERATOR, F"
        # "PEDESTRIAN, M"
        # "PEDESTRIAN, F"
        # "PASSENGER, M"
        # "PASSENGER, F"                
        # "BICYCLIST, M"                  
        # "MOTORCYCLE: PASSENGER, F"      
        # "MOTORCYCLE: OPERATOR, F"       
        # "BICYCLIST, F"                  
        # "OTHER, M"                      
        # "OTHER, F"                       
        
        writer.writerow([year, month, day, hour, minute, am_pm, locs[i], type_1, gender])

new_df = pd.read_csv('mass_fatality.csv')
print(new_df.head())

a = new_df['type'].value_counts()
print(a)
        
