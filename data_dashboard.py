# streamlit run data_dashboard.py

import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import KNeighborsClassifier
import shutup
import seaborn as sns

import collect_blast
import collect_yakker

shutup.please()

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

scrimmage = pd.read_csv('data/scrimmage_yakker/scrimmage_yakker.csv')
scrimmage['category'] = 'Scrimmage'

game = pd.read_csv('data/game_yakker/game_yakker.csv')
game['category'] = 'Game'

hitting_yakker = pd.concat([hitting, pitching, scrimmage, game], axis=0).drop_duplicates()
hitting_yakker = pd.merge(hitting_yakker, pd.DataFrame(player), left_on='Batter', right_on='yakker_name', how='left').drop_duplicates(subset=['Time', 'Batter', 'Pitcher'])

pitching_yakker = pd.concat([pitching, scrimmage, game], axis=0).drop_duplicates()
pitching_yakker = pitching_yakker[pitching_yakker['PitcherTeam'] == 'Georgia tech']

## Get blast data
blast = pd.read_csv('data/blast/full_data.csv')
blast = pd.merge(blast, pd.DataFrame(player), left_on='Player', right_on='blast_name', how='left')

date_format = "%b %d, %Y %I:%M:%S %p"
blast['Date'] = pd.to_datetime(blast['Date'], format=date_format).dt.date

bases_dict = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4, "Out": 0}

def find_hitting_stats(player, hitting_yakker=hitting_yakker):

    cur_yakkertech = hitting_yakker[(hitting_yakker['Batter'] == player)]
    # cur_yakkertech['Date'] = pd.to_datetime(cur_yakkertech['Date'], errors='coerce')
    # cur_yakkertech['Date'] = cur_yakkertech['Date'].dt.strftime('%m/%d/%Y')

    
    # Ensure consistent date format
    def format_date(date_str):
        try:
            date = pd.to_datetime(date_str)
            return date.strftime('%m/%d/%Y')
        except:
            return date_str

    cur_yakkertech['Date'] = cur_yakkertech['Date'].apply(format_date)

    cur_yakkertech['PlayResult'] = cur_yakkertech['PlayResult'].replace('Error', 'Out')
    cur_yakkertech.drop_duplicates(subset=['Date', 'PitchNo', 'ExitSpeed', 'Angle', 'Direction', 'Distance'], keep='first', inplace=True)

    bip = cur_yakkertech[(cur_yakkertech['PlayResult'] != 'Sacrifice') & (cur_yakkertech['PitchCall'] != 'Foul')]

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

def find_pitching_stats(player, pitching_yakker=pitching_yakker):
    
    cur_yakkertech = pitching_yakker[(pitching_yakker['Pitcher'] == player) & (pitching_yakker['category'] == 'Scrimmage')]
    cur_yakkertech['Date'] = pd.to_datetime(cur_yakkertech['Date'], errors='coerce')
    cur_yakkertech['Date'] = cur_yakkertech['Date'].dt.strftime('%m/%d/%Y')

    cur_yakkertech['PlayResult'] = cur_yakkertech['PlayResult'].replace('Error', 'Out')

    cur_yakkertech.drop_duplicates(subset=['Date', 'Time', 'ExitSpeed', 'Angle', 'Direction', 'Distance'], keep='first', inplace=True)
    
    bip = cur_yakkertech[(cur_yakkertech['PlayResult'] != 'Sacrifice') & (cur_yakkertech['PitchCall'] != 'Foul')]
    
    if(bip.shape[0] == 0):
        return
    
    bip["ExitSpeed"] = bip["ExitSpeed"].astype(float)
    bip["Angle"] = bip["Angle"].astype(float)
    bip["Direction"] = bip["Direction"].astype(float)
    bip["Distance"] = bip["Distance"].astype(float)
    
    bip = bip[['Date', 'ExitSpeed', 'Angle', 'Direction', 'Distance', 'PlateLocSide', 'PlateLocHeight', 'PlayResult']]
    
    bip = bip.dropna()
    
    bip = bip.reset_index(drop=True)
    
    y_pred = rf.predict(bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']])

    for i in range(bip.shape[0]):
        if pd.isna(bip.loc[i, 'PlayResult']):
            st.write(bip.loc[i])
            bip.loc[i, 'Bases'] = y_pred[i]
        else:
            bip.loc[i, 'Bases'] = bases_dict[bip.loc[i, 'PlayResult']]
    
    k = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Strikeout')].shape[0]
    bb = cur_yakkertech[(cur_yakkertech['KorBB'] == 'Walk')].shape[0] + cur_yakkertech[(cur_yakkertech['PitchCall'] == 'HitByPitch')].shape[0]
    
    expected_stats = pd.DataFrame({'BF': y_pred.shape[0] + k + bb, 'H': y_pred[y_pred != 0].shape[0], 'K': k, 'BB': bb, '2B': [np.count_nonzero(y_pred == 2)],
                                    '3B': [np.count_nonzero(y_pred == 3)], 'HR': [np.count_nonzero(y_pred == 4)]})

    
    expected_stats['outs'] = expected_stats['BF'] - expected_stats['H'] - expected_stats['BB']
    expected_stats['IP'] = expected_stats['outs'] / 3
    expected_stats['K/7'] = expected_stats['K'] / expected_stats['IP'] * 7
    expected_stats['BB/7'] = expected_stats['BB'] / expected_stats['IP'] * 7
    expected_stats['OPP AVG'] = expected_stats['H'] / expected_stats['BF']
    expected_stats['OBP'] = (expected_stats['H'] + expected_stats['BB']) / (expected_stats['BF'] + expected_stats['BB'])
    expected_stats['SLG'] = (expected_stats['H'] + 2 * expected_stats['2B'] + 3 * expected_stats['3B'] + 4 * expected_stats['HR']) / expected_stats['BF']
    expected_stats['OPP OPS'] = expected_stats['OBP'] + expected_stats['SLG']
    expected_stats['OPP wOBA'] = (0.69 * expected_stats['BB'] + 0.88 * (expected_stats['H'] - expected_stats['2B'] - expected_stats['3B'] - expected_stats['HR']) + 1.27 * expected_stats['2B'] + 1.62 * expected_stats['3B'] + 2.1 * expected_stats['HR']) / (expected_stats['BF'] + expected_stats['BB'] + expected_stats['K'])

    expected_stats = expected_stats[['BF', 'H', 'K', 'BB', 'HR', 'K/7', 'BB/7', 'OPP wOBA']]
    
    expected_stats = expected_stats.round(3)
    expected_stats[['OPP wOBA']] = expected_stats[['OPP wOBA']].applymap(lambda x: f'{x:.3f}')
    expected_stats[['K/7', 'BB/7']] = expected_stats[['K/7', 'BB/7']].applymap(lambda x: f'{round(x,1)}')

    
    return expected_stats


hitting_stats_df = pd.DataFrame(columns=['player', 'AB', 'H', 'K', 'BB', '2B', 'HR', 'AVG', 'SLG', 'OPS', 'wOBA'])
pitching_stats_df = pd.DataFrame(columns=['player', 'BF', 'H', 'K', 'BB', 'HR', 'K/7', 'BB/7', 'OPP wOBA'])
bip = pd.DataFrame()

unique_hitters = player['yakker_name']
unique_hitters = sorted(unique_hitters)

unique_pitchers = pitching_yakker['Pitcher'].unique()
unique_pitchers = unique_pitchers[~pd.isna(unique_pitchers)]
unique_pitchers = sorted(unique_pitchers)

hitting_tab, pitching_tab = st.tabs(['Hitting', 'Pitching'])

pitching_tab.title("2024-25 Georgia Tech Data Dashboard")
hitting_tab.title("2024-25 Georgia Tech Data Dashboard")

categories = hitting_tab.multiselect(
    'Select a category:',
    ['BP', 'Scrimmage', 'Game'], 
    default=['BP', 'Scrimmage', 'Game']
)

hitting_yakker = hitting_yakker[hitting_yakker['category'].isin(categories)]

for i in range(len(unique_hitters)):

    output = find_hitting_stats(unique_hitters[i], hitting_yakker=hitting_yakker)

    if output is None:
        continue
    
    cur_stats, cur_bip = output

    player_name = unique_hitters[i]
    cur_stats['player'] = player_name

    hitting_stats_df = pd.concat([hitting_stats_df, pd.DataFrame(cur_stats)], axis=0)
    bip = pd.concat([bip, cur_bip], axis=0)

for i in range(len(unique_pitchers)):
        
    output = find_pitching_stats(unique_pitchers[i], pitching_yakker=pitching_yakker)
    
    if output is None:
        continue
    
    cur_stats = output
    
    player_name = unique_pitchers[i]
    cur_stats['player'] = player_name
    
    pitching_stats_df = pd.concat([pitching_stats_df, pd.DataFrame(cur_stats)], axis=0)


hitting_stats_df = hitting_stats_df.reset_index(drop=True).sort_values(by='wOBA', ascending=False)
pitching_stats_df = pitching_stats_df.reset_index(drop=True).sort_values(by='OPP wOBA', ascending=True)

hitting_tab.markdown(f"<h3 style='text-align: center;'>Preseason Hitting Stats</h3>", unsafe_allow_html=True)

hitting_tab.dataframe(hitting_stats_df, use_container_width=True)
hitting_tab.write('')
hitting_tab.write('')
hitting_tab.write('')

pitching_tab.markdown(f"<h3 style='text-align: center;'>Preseason Pitching Stats</h3>", unsafe_allow_html=True)

pitching_tab.dataframe(pitching_stats_df, use_container_width=True)
pitching_tab.write('')
pitching_tab.write('')
pitching_tab.write('')

selected_hitter = hitting_tab.selectbox('Select a player:', [""] + list(unique_hitters), index=0)

selected_pitcher = pitching_tab.selectbox('Select a player:', [""] + list(unique_pitchers), index=0)

hitting_tab.markdown(f"<h3 style='text-align: center;'>{selected_hitter} Batted Ball Data</h3>", unsafe_allow_html=True)

bip = bip[bip['Batter'] == selected_hitter]
bip['PlayResult'] = bip['PlayResult'].fillna(bip['Bases'].map({v: k for k, v in bases_dict.items()}))
bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']] = bip[['ExitSpeed', 'Angle', 'Direction', 'Distance']].round(1)

hitting_tab.dataframe(bip.drop(columns=['Batter', 'Bases', 'PlateLocSide', 'PlateLocHeight']).sort_values(by='ExitSpeed', ascending=False).reset_index(drop=True), use_container_width=True)
hitting_tab.write('')
hitting_tab.write('')
hitting_tab.write('')

hitting_tab.markdown(f"<h3 style='text-align: center;'>{selected_hitter} Blast Data</h3>", unsafe_allow_html=True)

pitching_tab.markdown(f"<h3 style='text-align: center;'>{selected_pitcher} Pitch Profiles</h3>", unsafe_allow_html=True)

blast_player = blast[blast['yakker_name'] == selected_hitter]
blast_player['week'] = pd.to_datetime(blast_player['Date']).dt.to_period('W').dt.to_timestamp()
blast_player_agg = blast_player.groupby('week').agg({'Bat Speed (mph)': 'mean', 'Attack Angle (deg)': 'mean'}).reset_index()
blast_player_agg['Swings'] = blast_player.groupby('Date').size().reset_index(name='Count')['Count']

date_format = mdates.DateFormatter('%b %d, %Y')
blast_player_agg['week'] = blast_player_agg['week'].dt.strftime('%b %d, %Y')

blast_player_agg[['Bat Speed (mph)', 'Attack Angle (deg)']] = blast_player_agg[['Bat Speed (mph)', 'Attack Angle (deg)']].round(1)

hitting_tab.dataframe(blast_player_agg, use_container_width=True)
hitting_tab.download_button(
    label="Download Full Blast Data",
    data=blast_player.to_csv(index=False),
    file_name=f"{selected_hitter.lower().strip()} blast data.csv",
    mime="text/csv"
)

# make a small line graph of bat speed over time
fig, ax = plt.subplots()
ax.plot(blast_player_agg['week'], blast_player_agg['Bat Speed (mph)'], marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Bat Speed (mph)')
ax.set_title(f'{selected_hitter} Bat Speed Over Time')
plt.xticks(rotation=45)
plt.tight_layout()

hitting_tab.pyplot(fig) 

if selected_pitcher != "":

    selected_pitching_data = pitching_yakker[pitching_yakker['Pitcher'] == selected_pitcher].dropna(subset=['HorzBreak', 'InducedVertBreak', 'SpinAxis'])

    selected_pitching_data['Date'] = pd.to_datetime(selected_pitching_data['Date'], format='mixed', dayfirst=False)

    selected_pitching_data['date_display'] = selected_pitching_data.apply(
        lambda row: f"{row['Date'].strftime('%m/%d')}{' - ' + row['BatterTeam'] if row['category'] == 'Game' and pd.notnull(row['BatterTeam']) else ''}",
        axis=1
    )

    dates = selected_pitching_data['date_display'].unique()
    dates = sorted(dates)

    selected_dates = pitching_tab.multiselect('Select dates:', dates)
    select_all_dates_button = pitching_tab.checkbox("Select All")
    if select_all_dates_button:
        selected_dates = dates

    # Convert selected_dates back to regular format
    if selected_dates:
        selected_dates = [date.split(' - ')[0] for date in selected_dates]
        selected_dates = pd.to_datetime(selected_dates, format='%m/%d').strftime('%m/%d/2024')

        dates_fmt = [date.split(' - ')[0] for date in selected_dates]

        if len(selected_pitching_data[selected_pitching_data['TaggedPitchType'].notna()]) == 0:
            pitching_tab.write(f"Not enough data")
        else:

            for pitch in selected_pitching_data['TaggedPitchType'].unique():
                pitches = selected_pitching_data[selected_pitching_data['TaggedPitchType'] == pitch]

                avg_horz = pitches['HorzBreak'].mean()
                avg_vert = pitches['InducedVertBreak'].mean()

                # if horz is more than 3 std away from the mean, remove it
                pitches = pitches[(pitches['HorzBreak'] - avg_horz).abs() < 2 * pitches['HorzBreak'].std()]
                pitches = pitches[(pitches['InducedVertBreak'] - avg_vert).abs() < 2 * pitches['InducedVertBreak'].std()]

                selected_pitching_data = pd.concat([selected_pitching_data[selected_pitching_data['TaggedPitchType'] != pitch], pitches], axis=0)

            radians = np.deg2rad(selected_pitching_data['SpinAxis'])

            # Transform using sine and cosine
            selected_pitching_data['sin_axis'] = np.sin(radians)
            selected_pitching_data['cos_axis'] = np.cos(radians)

            if len(selected_pitching_data[selected_pitching_data['TaggedPitchType'].isna()]) != 0:

                train = selected_pitching_data[~selected_pitching_data['TaggedPitchType'].isna()]
                train['predicted'] = False

                # create a knn model to predict pitch type
                knn = KNeighborsClassifier(n_neighbors=len(train['TaggedPitchType'].unique()))
                knn.fit(train[['RelSpeed', 'HorzBreak', 'InducedVertBreak', 'sin_axis', 'cos_axis']], train['TaggedPitchType'])

                test = selected_pitching_data[selected_pitching_data['TaggedPitchType'].isna()]
                test['TaggedPitchType'] = knn.predict(test[['RelSpeed', 'HorzBreak', 'InducedVertBreak', 'sin_axis', 'cos_axis']])
                test['predicted'] = True

                selected_pitching_data = pd.concat([train, test], axis=0)

            selected_pitching_data = selected_pitching_data[selected_pitching_data['Date'].isin(dates_fmt)]

            # plot horzbreak vs induced vert break
            fig, ax = plt.subplots()
            for pitch in selected_pitching_data['TaggedPitchType'].unique():
                data = selected_pitching_data[(selected_pitching_data['TaggedPitchType'] == pitch)]
                ax.scatter(data['HorzBreak'], data['InducedVertBreak'], label=pitch, alpha=0.8)
            ax.set_xlabel('Horizontal Break (In.)')
            ax.set_ylabel('Induced Vertical Break (In.)')
            ax.legend()
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.axvline(0, color='black', linestyle='--', alpha=0.2)
            ax.axhline(0, color='black', linestyle='--', alpha=0.2)
            plt.tight_layout()
            pitching_tab.pyplot(fig)


            pitch_count = selected_pitching_data.groupby('TaggedPitchType').size().reset_index(name='count')

            pitch_data = selected_pitching_data.groupby('TaggedPitchType').agg({
                'RelSpeed': 'mean',
                'HorzBreak': 'mean',
                'InducedVertBreak': 'mean',
                'cos_axis': 'mean',
                'sin_axis': 'mean'
            }).reset_index()

            pitch_data = pitch_data.merge(pitch_count, on='TaggedPitchType')

            pitch_data = pitch_data.sort_values(by='count', ascending=False).reset_index(drop=True)

            pitch_data['SpinAxis'] = (np.rad2deg(np.arctan2(pitch_data['sin_axis'], pitch_data['cos_axis']))) % 360

            pitch_data['Tilt'] = pitch_data['SpinAxis'].apply(lambda x: x / 30 % 12)

            pitch_data['Tilt'] = pitch_data['Tilt'].apply(lambda x: f'{(int(x) + 6) % 12}:{int(x % 1 * 60):02d}')

            pitch_data['Tilt'] = pitch_data['Tilt'].replace(0, 12)

            pitch_data = pitch_data.round(1)
            pitch_data = pitch_data[['TaggedPitchType', 'count', 'RelSpeed', 'HorzBreak', 'InducedVertBreak', 'Tilt']]
            pitch_data.columns = ['Pitch Type', 'Count', 'Avg Speed (mph)', 'Avg Horz Break (in)', 'Avg Vert Break (in)', 'Spin Tilt']

            pitching_tab.dataframe(pitch_data, use_container_width=True)

            fig, ax = plt.subplots()
            for pitch_type in selected_pitching_data['TaggedPitchType'].unique():
                pitch_data = selected_pitching_data[selected_pitching_data['TaggedPitchType'] == pitch_type]
                ax.scatter(pitch_data['PlateLocSide'], pitch_data['PlateLocHeight'], s=50, alpha=0.8, label=pitch_type)
            sns.kdeplot(data=pitch_data, x='PlateLocSide', y='PlateLocHeight', levels=4, cmap='coolwarm', fill=True, alpha=0.2, linewidths=0, ax=ax)
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.plot([-17/24, 17/24], [1.5, 1.5], color='black')  # Bottom of strike zone
            ax.plot([-17/24, 17/24], [3.5, 3.5], color='black')  # Top of strike zone
            ax.plot([-17/24, -17/24], [1.5, 3.5], color='black')  # Left of strike zone
            ax.plot([17/24, 17/24], [1.5, 3.5], color='black')  # Right of strike zone
            ax.set_title(f'{selected_pitcher} - All Pitches', fontsize=10)
            ax.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            pitching_tab.pyplot(fig)
