import streamlit as st
import pandas as pd
import pymysql
import os
import altair as alt
import ssl
import pickle
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
load_dotenv()

rf = pickle.load(open("data_dashboard/xBases_model.pkl", "rb"))

cafile = os.getenv("SSL_CA_FILE_PATH")
capath = os.getenv("SSL_CA_PATH")

# Establish a MySQL connection with SSL/TLS configuration
connection = pymysql.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USERNAME"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    ssl=ssl.create_default_context(cafile=cafile, capath=capath)
)

query = """
SELECT player.id, blast.player, date, bat_speed, attack_angle, time_to_contact, power
FROM blast
INNER JOIN player ON blast.player = player.player_blast
"""

# Read data from the database
blast = pd.read_sql(query, connection, parse_dates=['date'])

query = """
SELECT player.id, yakkertech.Batter, Pitcher, Date, ExitSpeed, Angle, Direction, Distance, PitchCall, PlayResult, PlateLocSide, PlateLocHeight, KorBB, HorzBreak, InducedVertBreak, RelSpeed, SpinAxis
FROM yakkertech
INNER JOIN player ON yakkertech.Batter = player.player_yakker
"""

# Read data from the database
yakker = pd.read_sql(query, connection)

query = """
SELECT *
FROM player
"""

# Read data from the database
player = pd.read_sql(query, connection)

# Close the database connection
connection.close()

# Convert 'date' column to datetime format
blast['date'] = pd.to_datetime(blast['date'], format='%m/%d/%y')

pitching_tab, hitting_tab = st.tabs(["Pitching", "Hitting"])

pitching_tab.title("2023-24 Georgia Tech Data Dashboard")
hitting_tab.title("2023-24 Georgia Tech Data Dashboard")

# sort players by player_id
blast = blast.sort_values(['id'])

hitting_tab.subheader("Preseason Stats")

bases_dict = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4, "Out": 0}

def find_stats(id):

    cur_yakkertech = yakker[(yakker['id'] == id)]

    bip = cur_yakkertech[(~cur_yakkertech['ExitSpeed'].isna()) & (cur_yakkertech['Direction'] > -45) & (cur_yakkertech['Direction'] < 45)]

    if(bip.shape[0] == 0):
        return

    bip["ExitSpeed"] = bip["ExitSpeed"].astype(float)
    bip["Angle"] = bip["Angle"].astype(float)
    bip["Direction"] = bip["Direction"].astype(float)
    bip["Distance"] = bip["Distance"].astype(float)

    bip = bip[['ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlateLocSide', 'PlateLocHeight', 'PlayResult', 'Date']]

    bip = bip.dropna()

    bip = bip.reset_index(drop=True)

    y_pred = rf.predict(bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']])

    for i in range(bip.shape[0]):
        if bip.loc[i, 'Date'] >= "2024-01-13":
            bip.loc[i, 'Bases'] = bases_dict[bip.loc[i, 'PlayResult']]
        else:
            bip.loc[i, 'Bases'] = y_pred[i]

    k = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Strikeout')].shape[0]
    bb = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Walk')].shape[0] + cur_yakkertech[(cur_yakkertech['PitchCall'] == 'HitByPitch')].shape[0]

    expected_stats = pd.DataFrame({'AB': y_pred.shape[0] + k + bb, 'H': y_pred[y_pred != 0].shape[0], 'K': k, 'BB': bb, '2B': [np.count_nonzero(y_pred == 2)],
                                    '3B': [np.count_nonzero(y_pred == 3)], 'HR': [np.count_nonzero(y_pred == 4)]})

    expected_stats['AVG'] = expected_stats['H'] / expected_stats['AB']
    expected_stats['OBP'] = (expected_stats['H'] + expected_stats['BB']) / (expected_stats['AB'] + expected_stats['BB'])
    expected_stats['SLG'] = (expected_stats['H'] + 2 * expected_stats['2B'] + 3 * expected_stats['3B'] + 4 * expected_stats['HR']) / expected_stats['AB']
    expected_stats['OPS'] = expected_stats['OBP'] + expected_stats['SLG']
    expected_stats['wOBA'] = (0.69 * expected_stats['BB'] + 0.88 * (expected_stats['H'] - expected_stats['2B'] - expected_stats['3B'] - expected_stats['HR']) + 1.27 * expected_stats['2B'] + 1.62 * expected_stats['3B'] + 2.1 * expected_stats['HR']) / (expected_stats['AB'] + expected_stats['BB'] + expected_stats['K'])

    expected_stats = expected_stats.round(3)
    expected_stats[['AVG', 'SLG', 'OPS', 'wOBA']] = expected_stats[['AVG', 'SLG', 'OPS', 'wOBA']].applymap(lambda x: f'{x:.3f}')
    expected_stats = expected_stats[['AB', 'H', 'K', 'BB', '2B', 'HR', 'AVG', 'SLG', 'OPS', 'wOBA']]

    return expected_stats

stats_df = pd.DataFrame(columns=['player', 'AB', 'H', 'K', 'BB', '2B', 'HR', 'AVG', 'SLG', 'OPS', 'wOBA'])

for i in range(player.shape[0]):
    cur_stats = pd.DataFrame(find_stats(player.loc[player.index[i], 'id']))

    player_name = player.loc[player.index[i], 'player_blast']
    cur_stats['player'] = player_name


    stats_df = pd.concat([stats_df, pd.DataFrame(cur_stats)], axis=0)

stats_df = stats_df.reset_index(drop=True)

stats_df.to_csv('data_dashboard/expected_hitting_stats.csv', index=False)

hitting_tab.dataframe(stats_df, width=700, hide_index=True)


# Create a dropdown widget for player selection
selected_player = hitting_tab.selectbox("Select Player", blast['player'].unique())
selected_player_id = player[player['player_blast'] == selected_player]['id'].values[0]

### Game Stats ###

table = pd.read_csv('https://softballstatline.com/teams/data/hitting_stats/hitting_stats_2023.csv')

name_dict = {'Sara beth Allen': 'SB Allen', 'Sandra beth Pritchett': 'SB Pritchett'}

table['player'] = table['player'].replace(name_dict)

gt_stats = table[(table['player'] == selected_player)]
gt_stats = gt_stats[['ab', 'h', 'x2b', 'x3b', 'hr', 'rbi', 'avg', 'obp', 'ops']]
gt_stats = gt_stats.rename(columns={'ab': 'AB', 'h': 'H', 'x2b': '2B', 'x3b': '3B', 'hr': 'HR', 'rbi': 'RBI', 'avg': 'AVG', 'obp': 'OBP', 'ops': 'OPS'})

hitting_tab.subheader("2023 Game Stats")

if gt_stats.empty:
    hitting_tab.write("No stats available for this player")
else:
    hitting_tab.dataframe(gt_stats, hide_index=True, width=700)

### Expected Stats Prediction ###
hitting_tab.subheader("Preseason Stats")

cur_yakkertech = yakker[(yakker['id'] == selected_player_id)]

bases_dict = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4, "Out": 0}

bip = cur_yakkertech[(~cur_yakkertech['ExitSpeed'].isna()) & (cur_yakkertech['Direction'] > -45) & (cur_yakkertech['Direction'] < 45)]

bip["ExitSpeed"] = bip["ExitSpeed"].astype(float)
bip["Angle"] = bip["Angle"].astype(float)
bip["Direction"] = bip["Direction"].astype(float)
bip["Distance"] = bip["Distance"].astype(float)

bip = bip[['ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlateLocSide', 'PlateLocHeight']]

bip = bip.dropna()

bip = bip.reset_index(drop=True)

y_pred = rf.predict(bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']])
bip['Bases'] = y_pred

k = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Strikeout')].shape[0]
bb = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Walk')].shape[0]

expected_stats = pd.DataFrame({'AB': y_pred.shape[0] + k + bb, 'H': y_pred[y_pred != 0].shape[0], 'K': k, 'BB': bb, '2B': [np.count_nonzero(y_pred == 2)],
                                 '3B': [np.count_nonzero(y_pred == 3)], 'HR': [np.count_nonzero(y_pred == 4)]})

expected_stats['AVG'] = expected_stats['H'] / expected_stats['AB']
expected_stats['OBP'] = (expected_stats['H'] + expected_stats['BB']) / (expected_stats['AB'] + expected_stats['BB'])
expected_stats['SLG'] = (expected_stats['H'] + 2 * expected_stats['2B'] + 3 * expected_stats['3B'] + 4 * expected_stats['HR']) / expected_stats['AB']
expected_stats['OPS'] = expected_stats['OBP'] + expected_stats['SLG']

expected_stats = expected_stats.round(3)

hitting_tab.dataframe(expected_stats, width=700, hide_index=True)

count_dict = {}

# Iterate through the list and count occurrences of each unique value
for num in y_pred:
    if num in count_dict:
        count_dict[num] += 1
    else:
        count_dict[num] = 1

# Create a DataFrame from the count_dict
counts = pd.DataFrame(list(count_dict.items()), columns=["Value", "Count"])

### Progression over Time ####

hitting_tab.header("Progression over Time")

# Create a radio button for selecting data source
selected_data_source = hitting_tab.radio("Select Data Source", ['Blast Data', 'Yakkertech Data'])

# Modify the query based on the selected data source
if selected_data_source == 'Blast Data':
    selected_data_columns = ['bat_speed', 'attack_angle', 'time_to_contact', 'power']
else:
    # Modify the query or add another query for Yakkertech data
    selected_data_columns = ['ExitSpeed', 'Distance']

# Create a dropdown widget for selecting the specific data
selected_data_point = hitting_tab.selectbox("Select Data", selected_data_columns)


def convert_to_date(date_str):
    try:
        # First, try to parse in MM/DD/YYYY format
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        # If it fails, parse in MM/DD/YY format
        return pd.to_datetime(date_str, format='%m/%d/%y')

# Filter the data for the selected player
if selected_data_source == 'Blast Data':
    selected_player_data = blast[blast['player'] == selected_player]
    selected_data = selected_player_data[['player', 'date', selected_data_point]]

else:
    selected_player_data = yakker[yakker['id'] == selected_player_id]
    selected_data = selected_player_data[['Batter', 'Date', selected_data_point]]
    selected_data = selected_data.rename(columns={'Date': 'date', 'Batter': 'player'})
    selected_data['date'] = selected_data['date'].apply(convert_to_date)

selected_data['date'] = [date.strftime('%m/%d/%Y') for date in selected_data['date']]

if selected_data_point == "attack_angle":
    selected_data = selected_data[selected_data['attack_angle'] != -1000]
elif (selected_data_point == "time_to_contact") | (selected_data_point == "power"):
    selected_data = selected_data[selected_data[selected_data_point] != -1]

# Group the data by date and calculate statistics
stats_by_date = selected_data.groupby(selected_data['date'])[selected_data_point].describe(percentiles=[0.25, 0.5, 0.75])

# Aggregations for specific columns
total_count = stats_by_date['count'].sum()
min_of_min = stats_by_date['min'].min()
max_of_max = stats_by_date['max'].max()

# Mean for other columns (excluding 'count', 'min', 'max')
mean_values = stats_by_date.drop(columns=['count', 'min', 'max']).mean()

# Combine into a new row
total_row = pd.Series([total_count, min_of_min, max_of_max] + list(mean_values), 
                      index=['count', 'min', 'max'] + list(mean_values.index))

# Append the total row to the existing DataFrame
stats_by_date.loc['Total'] = total_row

stats_by_date['mean'] = stats_by_date['mean'].round(2)
stats_by_date['std'] = stats_by_date['std'].round(2)
stats_by_date['25%'] = stats_by_date['25%'].round(2)
stats_by_date['50%'] = stats_by_date['50%'].round(2)
stats_by_date['75%'] = stats_by_date['75%'].round(2)

# Display the updated table
hitting_tab.dataframe(stats_by_date.dropna(axis=0), width=1000)

cur_yakkertech['swing'] = cur_yakkertech['PitchCall'].isin(['StrikeSwinging', 'Foul', 'InPlay'])

cur_yakkertech['swing_middle'] = (cur_yakkertech['PlateLocSide'] > -17/48) & (cur_yakkertech['PlateLocSide'] < 17/48) & (cur_yakkertech['PlateLocHeight'] > 18/9) & (cur_yakkertech['PlateLocHeight'] < 42/14) & (cur_yakkertech['swing'] == True)
cur_yakkertech['swing_strike'] = (cur_yakkertech['PlateLocSide'] > -17/24) & (cur_yakkertech['PlateLocSide'] < 17/24) & (cur_yakkertech['PlateLocHeight'] > 18/12) & (cur_yakkertech['PlateLocHeight'] < 42/12) & (cur_yakkertech['swing'] == True) & (cur_yakkertech['swing_middle'] == False)

cur_yakkertech['swing_type'] = cur_yakkertech.apply(lambda x: 'Swing Middle' if x['swing_middle'] else 'Swing Strike' if x['swing_strike'] else 'Swing Ball' if x['swing'] else 'No Swing', axis=1)
cur_yakkertech['swing_type_color'] = cur_yakkertech.apply(lambda x: 'green' if x['swing_middle'] else 'yellow' if x['swing_strike'] else 'red' if x['swing'] else 'black', axis=1)

plt.scatter(cur_yakkertech['PlateLocSide'], cur_yakkertech['PlateLocHeight'], c=cur_yakkertech['swing_type_color'], alpha=0.75)
plt.legend(['Green', 'Yellow', 'Red', 'Black'], ['Middle', 'Strike', 'Ball', 'No Swing'], loc='upper right')

plt.plot([-17/24, 17/24], [18/12, 18/12], color='black')
plt.plot([-17/24, 17/24], [42/12, 42/12], color='black')
plt.plot([-17/24, -17/24], [18/12, 42/12], color='black')
plt.plot([17/24, 17/24], [18/12, 42/12], color='black')

plt.plot([-17/48, 17/48], [18/9, 18/9], color='black', linestyle='dashed')
plt.plot([-17/48, 17/48], [42/14, 42/14], color='black', linestyle='dashed')
plt.plot([-17/48, -17/48], [18/9, 42/14], color='black', linestyle='dashed')
plt.plot([17/48, 17/48], [18/9, 42/14], color='black', linestyle='dashed')

plt.plot([29/24, 29/24], [-1, 6], color='black', linestyle='dotted')
plt.plot([-29/24, -29/24], [-1, 6], color='black', linestyle='dotted')
plt.xlim(-3, 3)
plt.ylim(-1, 6)
hitting_tab.pyplot(use_container_width=True)

unique_pitchers = yakker['Pitcher'].unique()
unique_pitchers = [pitcher for pitcher in unique_pitchers if pitcher not in ["Emery Bales", "Sb Allen", None]]

pitching_tab.subheader("Preseason Stats")

def find_pitching_stats(id):
    
    cur_yakkertech = yakker[(yakker['Pitcher'] == id)]
    
    bip = cur_yakkertech[(~cur_yakkertech['ExitSpeed'].isna()) & (cur_yakkertech['Direction'] > -45) & (cur_yakkertech['Direction'] < 45)]
    
    if(bip.shape[0] == 0):
        return
    
    bip["ExitSpeed"] = bip["ExitSpeed"].astype(float)
    bip["Angle"] = bip["Angle"].astype(float)
    bip["Direction"] = bip["Direction"].astype(float)
    bip["Distance"] = bip["Distance"].astype(float)
    
    bip = bip[['ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlateLocSide', 'PlateLocHeight']]
    
    bip = bip.dropna()
    
    bip = bip.reset_index(drop=True)
    
    y_pred = rf.predict(bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']])
    bip['Bases'] = y_pred
    
    k = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Strikeout')].shape[0]
    bb = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Walk')].shape[0] + cur_yakkertech[(cur_yakkertech['PitchCall'] == 'HitByPitch')].shape[0]
    
    expected_stats = pd.DataFrame({'AB': y_pred.shape[0] + k + bb, 'H': y_pred[y_pred != 0].shape[0], 'K': k, 'BB': bb, '2B': [np.count_nonzero(y_pred == 2)],
                                    '3B': [np.count_nonzero(y_pred == 3)], 'HR': [np.count_nonzero(y_pred == 4)]})
    
    expected_stats['OPP AVG'] = expected_stats['H'] / expected_stats['AB']
    expected_stats['OBP'] = (expected_stats['H'] + expected_stats['BB']) / (expected_stats['AB'] + expected_stats['BB'])
    expected_stats['SLG'] = (expected_stats['H'] + 2 * expected_stats['2B'] + 3 * expected_stats['3B'] + 4 * expected_stats['HR']) / expected_stats['AB']
    expected_stats['OPP OPS'] = expected_stats['OBP'] + expected_stats['SLG']
    expected_stats['BF'] = expected_stats['AB'] + expected_stats['BB'] + expected_stats['K']
    expected_stats['OPP wOBA'] = (0.69 * expected_stats['BB'] + 0.88 * (expected_stats['H'] - expected_stats['2B'] - expected_stats['3B'] - expected_stats['HR']) + 1.27 * expected_stats['2B'] + 1.62 * expected_stats['3B'] + 2.1 * expected_stats['HR']) / (expected_stats['AB'] + expected_stats['BB'] + expected_stats['K'])

    expected_stats = expected_stats[['BF', 'H', 'K', 'BB', 'HR', 'OPP AVG', 'OPP OPS', 'OPP wOBA']]
    
    expected_stats = expected_stats.round(3)
    expected_stats[['OPP AVG', 'OPP OPS', 'OPP wOBA']] = expected_stats[['OPP AVG', 'OPP OPS', 'OPP wOBA']].applymap(lambda x: f'{x:.3f}')
    
    return expected_stats

pitcher_stats_df = pd.DataFrame(columns=['Pitcher', 'BF', 'H', 'K', 'BB', 'HR', 'OPP AVG', 'OPP OPS', 'OPP wOBA'])

for i in range(len(unique_pitchers)):
    cur_stats = pd.DataFrame(find_pitching_stats(unique_pitchers[i]))
    
    pitcher_name = unique_pitchers[i]
    cur_stats['Pitcher'] = pitcher_name
    
    pitcher_stats_df = pd.concat([pitcher_stats_df, pd.DataFrame(cur_stats)], axis=0)

pitching_tab.dataframe(pitcher_stats_df, width=700, hide_index=True)

selected_pitcher = pitching_tab.selectbox("Select Pitcher", unique_pitchers)
all_dates = yakker[yakker['Pitcher'] == selected_pitcher]['Date'].unique()
selected_dates = pitching_tab.multiselect("Select Dates", ['All'] + list(all_dates), default=['All'])
num_pitches = pitching_tab.slider("Number of Pitches", 2, 5, 1)

if 'All' in selected_dates:
    cur_pitching = yakker[(yakker['Pitcher'] == selected_pitcher) & (yakker['HorzBreak'] < 15) & (yakker['HorzBreak'] > -15) & (yakker['InducedVertBreak'] < 15) & (yakker['InducedVertBreak'] > -15)].dropna(subset=['HorzBreak', 'InducedVertBreak']).reset_index(drop=True)
else:
    cur_pitching = yakker[(yakker['Pitcher'] == selected_pitcher) & (yakker['HorzBreak'] < 15) & (yakker['HorzBreak'] > -15) & (yakker['InducedVertBreak'] < 15) & (yakker['InducedVertBreak'] > -15) & (yakker['Date'].isin(selected_dates))].dropna(subset=['HorzBreak', 'InducedVertBreak']).reset_index(drop=True)

radians = np.deg2rad(cur_pitching['SpinAxis'])

# Transform using sine and cosine
cur_pitching['sin_axis'] = np.sin(radians)
cur_pitching['cos_axis'] = np.cos(radians)

model = KMeans(n_clusters=num_pitches)

model.fit(cur_pitching[['HorzBreak', 'InducedVertBreak', 'RelSpeed', 'sin_axis', 'cos_axis']])

cur_pitching['Type'] = model.predict(cur_pitching[['HorzBreak', 'InducedVertBreak', 'RelSpeed', 'sin_axis', 'cos_axis']])

unique_labels = cur_pitching['Type'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
label_to_color = dict(zip(unique_labels, colors))

# Now plot using these colors
for label in unique_labels:
    subset = cur_pitching[cur_pitching['Type'] == label]
    plt.scatter(subset['HorzBreak'], subset['InducedVertBreak'], 
                color=label_to_color[label], label=label)

plt.legend()
plt.xlabel('Horizontal Break')
plt.ylabel('Induced Vertical Break')
plt.xlim(-15, 15)
plt.ylim(-15, 15)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Display the plot in Streamlit
pitching_tab.pyplot()

pitch_metrics = cur_pitching.groupby('Type', as_index=False)[['RelSpeed', 'HorzBreak', 'InducedVertBreak', 'SpinAxis']].mean()

def convert_to_clock(degrees):
    # Map the degree to the hour on a 12-hour clock
    # 180 degrees = 12:00, 0 degrees = 6:00
    hours = ((degrees - 180) % 360) / -30
    hours = (hours + 12) % 12  # Convert to 12-hour format
    
    # Convert fractional hours to minutes
    minutes = int((hours % 1) * 60)
    
    # Convert hours to integer
    hours = int(hours)

    # Format as a time string
    return f'{hours:02d}:{minutes:02d}'

# Example usage:
pitch_metrics['Tilt'] = pitch_metrics['SpinAxis'].apply(convert_to_clock)

pitching_tab.dataframe(pitch_metrics[['Type', 'RelSpeed', 'HorzBreak', 'InducedVertBreak', 'Tilt']], width=700, hide_index=True)

loc_plot = sns.kdeplot(data=cur_pitching, x='PlateLocSide', y='PlateLocHeight', fill=True, cmap='coolwarm', levels=8, thresh=0.05, alpha=0.5)
plt.plot([-17/24, 17/24], [18/12, 18/12], color='black')
plt.plot([-17/24, 17/24], [42/12, 42/12], color='black')
plt.plot([-17/24, -17/24], [18/12, 42/12], color='black')
plt.plot([17/24, 17/24], [18/12, 42/12], color='black')
plt.plot([29/24, 29/24], [-1, 6], color='black', linestyle='dotted')
plt.plot([-29/24, -29/24], [-1, 6], color='black', linestyle='dotted')
plt.xlim(-3, 3)
plt.ylim(-1, 6)

pitching_tab.pyplot()