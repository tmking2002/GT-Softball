import pandas as pd
import os

files = os.listdir('data/blast')
data = pd.DataFrame()

for file in files:
    if (file.endswith('.csv')) and (file != 'full_data.csv'):
        cur_data = pd.read_csv('data/blast/' + file, skiprows=8)
        player = file.removesuffix('.csv')

        cur_data['Player'] = player

        data = pd.concat([data, cur_data])

data.to_csv('data/blast/full_data.csv', index=False)