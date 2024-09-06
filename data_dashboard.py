# streamlit run data_dashboard.py

import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

import collect_blast
import collect_yakker

rf = pickle.load(open("xBases_model.pkl", "rb"))

player = {
    "blast_name": ['Addison Leschber', 'Alyssa Willer', 'Bri Condon', 'Camden Anders', 'Caroline Patterson', 'Eliana Gottlieb', 'Ella Edgmon', 'Emma Minghini', 'Emma Simon', 'Grace Connelly', 'Gracyn Tucker', 'Jayden Gailey', 'Kaya Booker', 'Lillian Martineau', 'Maddie Furniss', 'Makayla Coffield', 'Paige Vukadinovich', 'Reese Hunter'],
    "yakker_name": ['Addison Leschber', 'Alyssa Willer', 'Brionna Condon', 'Camden Anders', 'Caroline Patterson', 'Eliana Gottlieb', 'Ella Edgmon', 'Emma Minghini', 'Emma Simon', 'Grace Connelly', 'Gracyn Tucker', 'Jayden Gailey', 'Kaya Booker', 'Lillian Martineau', 'Maddie Furniss', 'Makayla Coffield', 'Paige Vukadinovich', 'Reese Hunter']
}


## Get yakker data
hitting = pd.read_csv('data/hitting_yakker/hitting_yakker.csv')
hitting['category'] = 'BP'

pitching = pd.read_csv('data/pitching_yakker/pitching_yakker.csv')
pitching['category'] = 'BP'

scrimmage_hitting = pd.read_csv('data/scrimmage_yakker/scrimmage_yakker.csv')
scrimmage_hitting['category'] = 'Scrimmage'

hitting_yakker = pd.concat([hitting, pitching, scrimmage_hitting], axis=0).drop_duplicates()
hitting_yakker = pd.merge(hitting_yakker, pd.DataFrame(player), left_on='Batter', right_on='yakker_name', how='left')

pitching_yakker = pd.concat([pitching, scrimmage_hitting], axis=0)

## Get blast data
blast = pd.read_csv('data/blast/full_data.csv')
blast = pd.merge(blast, pd.DataFrame(player), left_on='Player', right_on='blast_name', how='left')
blast['Date'] = pd.to_datetime(blast['Date']).dt.date

bases_dict = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4, "Out": 0}

def find_hitting_stats(player, hitting_yakker=hitting_yakker):

    cur_yakkertech = hitting_yakker[(hitting_yakker['Batter'] == player)]

    bip = cur_yakkertech[(~cur_yakkertech['ExitSpeed'].isna()) & (cur_yakkertech['Direction'] > -45) & (cur_yakkertech['Direction'] < 45)]

    if(bip.shape[0] == 0):
        return

    bip["ExitSpeed"] = bip["ExitSpeed"].astype(float)
    bip["Angle"] = bip["Angle"].astype(float)
    bip["Direction"] = bip["Direction"].astype(float)
    bip["Distance"] = bip["Distance"].astype(float)

    bip = bip[['Batter', 'Pitcher', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlateLocSide', 'PlateLocHeight', 'PlayResult', 'Date']]

    bip = bip.dropna(subset=['ExitSpeed', 'Angle', 'Direction', 'Distance'])

    bip = bip.reset_index(drop=True)

    y_pred = rf.predict(bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']])

    for i in range(bip.shape[0]):
        if pd.isna(bip.loc[i, 'PlayResult']):
            bip.loc[i, 'Bases'] = y_pred[i]
        else:
            bip.loc[i, 'Bases'] = bases_dict[bip.loc[i, 'PlayResult']]

    k = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Strikeout')].shape[0]
    bb = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Walk')].shape[0] + cur_yakkertech[(cur_yakkertech['PitchCall'] == 'HitByPitch')].shape[0]

    expected_stats = pd.DataFrame({
        'AB': [y_pred.shape[0] + k + bb], 
        'H': [len(bip.loc[bip['Bases'] != 0])], 
        'K': [k], 
        'BB': [bb], 
        '2B': [len(bip.loc[bip['Bases'] == 2])], 
        '3B': [len(bip.loc[bip['Bases'] == 3])], 
        'HR': [len(bip.loc[bip['Bases'] == 4])]
    })

    expected_stats['AVG'] = expected_stats['H'] / expected_stats['AB']
    expected_stats['OBP'] = (expected_stats['H'] + expected_stats['BB']) / (expected_stats['AB'] + expected_stats['BB'])
    expected_stats['SLG'] = (expected_stats['H'] + 2 * expected_stats['2B'] + 3 * expected_stats['3B'] + 4 * expected_stats['HR']) / expected_stats['AB']
    expected_stats['OPS'] = expected_stats['OBP'] + expected_stats['SLG']
    expected_stats['wOBA'] = (0.69 * expected_stats['BB'] + 0.88 * (expected_stats['H'] - expected_stats['2B'] - expected_stats['3B'] - expected_stats['HR']) + 1.27 * expected_stats['2B'] + 1.62 * expected_stats['3B'] + 2.1 * expected_stats['HR']) / (expected_stats['AB'] + expected_stats['BB'] + expected_stats['K'])

    expected_stats = expected_stats.round(3)
    expected_stats[['AVG', 'SLG', 'OPS', 'wOBA']] = expected_stats[['AVG', 'SLG', 'OPS', 'wOBA']].applymap(lambda x: f'{x:.3f}')
    expected_stats = expected_stats[['AB', 'H', 'K', 'BB', '2B', 'HR', 'AVG', 'SLG', 'OPS', 'wOBA']]

    if bip is None:
        bip = pd.DataFrame()
        
    return expected_stats, bip


stats_df = pd.DataFrame(columns=['player', 'AB', 'H', 'K', 'BB', '2B', 'HR', 'AVG', 'SLG', 'OPS', 'wOBA'])
bip = pd.DataFrame()

unique_hitters = hitting_yakker['Batter'].unique()
unique_hitters = unique_hitters[~pd.isna(unique_hitters)]
unique_hitters = unique_hitters[unique_hitters != 'Sara beth Allen']
unique_hitters = sorted(unique_hitters)

pitching_tab, hitting_tab = st.tabs(["Pitching", "Hitting"])

pitching_tab.title("2024-25 Georgia Tech Data Dashboard")
hitting_tab.title("2024-25 Georgia Tech Data Dashboard")

hitting_tab.subheader("Preseason Stats")

categories = hitting_tab.multiselect(
    'Select a category:',
    ['BP', 'Scrimmage'], 
    default=['BP', 'Scrimmage']
)

hitting_yakker = hitting_yakker[hitting_yakker['category'].isin(categories)]

for i in range(len(unique_hitters)):

    output = find_hitting_stats(unique_hitters[i], hitting_yakker=hitting_yakker)

    if output is None:
        continue
    
    cur_stats, cur_bip = output

    player_name = unique_hitters[i]
    cur_stats['player'] = player_name

    stats_df = pd.concat([stats_df, pd.DataFrame(cur_stats)], axis=0)
    bip = pd.concat([bip, cur_bip], axis=0)

stats_df = stats_df.reset_index(drop=True).sort_values(by='wOBA', ascending=False)

hitting_tab.markdown(f"<h3 style='text-align: center;'>Preseason Hitting Stats</h3>", unsafe_allow_html=True)

hitting_tab.dataframe(stats_df, use_container_width=True)
hitting_tab.write('')
hitting_tab.write('')
hitting_tab.write('')

selected_hitter = hitting_tab.selectbox('Select a player:', [""] + list(unique_hitters), index=0)

hitting_tab.markdown(f"<h3 style='text-align: center;'>{selected_hitter} Batted Ball Data</h3>", unsafe_allow_html=True)

bip = bip[bip['Batter'] == selected_hitter]
bip['PlayResult'] = bip['PlayResult'].fillna(bip['Bases'].map({v: k for k, v in bases_dict.items()}))
bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']] = bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']].round(1)

hitting_tab.dataframe(bip.drop(columns=['Batter', 'Bases', 'PlateLocSide', 'PlateLocHeight']).sort_values(by='ExitSpeed', ascending=False).reset_index(drop=True), use_container_width=True)
hitting_tab.write('')
hitting_tab.write('')
hitting_tab.write('')

hitting_tab.markdown(f"<h3 style='text-align: center;'>{selected_hitter} Blast Data</h3>", unsafe_allow_html=True)

blast_player = blast[blast['yakker_name'] == selected_hitter]
blast_player_agg = blast_player.groupby('Date').agg({'Bat Speed (mph)': 'mean', 'Attack Angle (deg)': 'mean'}).reset_index()
blast_player_agg['Swings'] = blast_player.groupby('Date').size().reset_index(name='Count')['Count']

blast_player_agg[['Bat Speed (mph)', 'Attack Angle (deg)']] = blast_player_agg[['Bat Speed (mph)', 'Attack Angle (deg)']].round(1)

hitting_tab.dataframe(blast_player_agg, use_container_width=True)
hitting_tab.download_button(
    label="Download Blast Data",
    data=blast_player.to_csv(index=False),
    file_name=f"{selected_hitter.lower().strip()} blast data.csv",
    mime="text/csv"
)

# make a small line graph of bat speed over time
fig, ax = plt.subplots()
ax.plot(blast_player_agg['Date'], blast_player_agg['Bat Speed (mph)'], marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Bat Speed (mph)')
ax.set_title(f'{selected_hitter} Bat Speed Over Time')
plt.xticks(rotation=45)

hitting_tab.pyplot(fig)