import pandas as pd
import os

# get pitching data

files = os.listdir('data/pitching_yakker')

data = pd.DataFrame()

for file in files:
    if (file.endswith('.csv')) and (file != 'pitching_yakker.csv'):
        cur_data = pd.read_csv('data/pitching_yakker/' + file)

        data = pd.concat([data, cur_data])

data.to_csv('data/pitching_yakker/pitching_yakker.csv', index=False)

# get hitting data

files = os.listdir('data/hitting_yakker')

data = pd.DataFrame()

for file in files:
    if (file.endswith('.csv')) and (file != 'hitting_yakker.csv'):
        cur_data = pd.read_csv('data/hitting_yakker/' + file)

        data = pd.concat([data, cur_data])

data.to_csv('data/hitting_yakker/hitting_yakker.csv', index=False)

# get scrimmage data

files = os.listdir('data/scrimmage_yakker')

data = pd.DataFrame()

for file in files:
    if (file.endswith('.csv')) and (file != 'scrimmage_yakker.csv'):
        cur_data = pd.read_csv('data/scrimmage_yakker/' + file)

        data = pd.concat([data, cur_data])

data.to_csv('data/scrimmage_yakker/scrimmage_yakker.csv', index=False)
