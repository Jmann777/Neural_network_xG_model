"""
The following script using the neural network created in Model.py and trained in xG_advancement.py to calculate
the xG for each player at Euro 2020. It then outputs the top 10 players with the highest xG
"""
import joblib
import os
import numpy as np
import Statsbomb as Sb
import Shots_Features_Sb as pdf
import tensorflow as tf
import random as rn
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import load_model

# Setting random seeds in order to allow for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

""" ***** Data preparation for the model *****"""
cid: int = 55
sid: int = 43
df_match: pd.DataFrame = Sb.sb_matches(cid, sid)
shots: pd.DataFrame = Sb.sb_shots_season(cid, sid)
track_df: pd.DataFrame = Sb.sb_tracking_season(cid, sid)

# Filtering out non open_play shots
shots = shots[shots["sub_type_name"] == "Open Play"]
# Filter out shots where goalkeeper was not tracked
gks_tracked: np.ndarray = track_df[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][
    'id'].unique()
shots = shots[shots["id"].isin(gks_tracked)]

# Model variables
model_vars: pd.DataFrame = pdf.default_model_vars(shots=shots)
b: pd.Series = pdf.params(model_vars)
model_vars["xg_basic"] = model_vars.apply(pdf.calculate_xG, b=b, axis=1)
model_vars["gk_distance"] = shots.apply(pdf.dist_to_gk, track_df=track_df, axis=1)
model_vars["gk_distance_y"] = shots.apply(pdf.y_to_gk, track_df=track_df, axis=1)
model_vars["close_players"] = shots.apply(pdf.three_meters_away, track_df=track_df, axis=1)
model_vars["triangle"] = shots.apply(pdf.players_in_triangle, track_df=track_df, axis=1)
model_vars["gk_dist_to_goal"] = shots.apply(pdf.gk_dist_to_goal, track_df=track_df, axis=1)
# create binary variable 1 if ball is closer to the goal than goalkeeper
model_vars["is_closer"] = np.where(model_vars["gk_dist_to_goal"] > model_vars["distance"], 1, 0)
# create binary variable 1 if header
model_vars["header"] = shots.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)

""" ***** Loading and using the model to predict the top 10 players with highest xG according to the model in
Euro 2020. ***** """
# load scaler details
scaler = joblib.load('fitted_scaler.joblib')
# load trained model
model = load_model('best_model.h5')

# Store model vars into a matrix
X_unseen: np.ndarray = model_vars[["x0", "is_closer", "angle", "distance",
                                   "gk_distance", "gk_distance_y",
                                   "triangle", "close_players", "header", "xg_basic"]].values
# Scale independent variables
X_unseen: np.ndarray = scaler.transform(X_unseen)
# Make calculation on euros data
xgs_euro: np.ndarray = model.predict(X_unseen)
# Assign xG to the shots dataframe
shots["our_xG"] = xgs_euro
# Output the top 10 players with the highest xG
print(shots.groupby(["player_name"])["our_xG"].sum().sort_values(ascending=False)[:10].reset_index())
# Group by player_name and sum the xG values
top_players_xg: pd.DataFrame = shots.groupby(["player_name"])["our_xG"].sum().sort_values(ascending=False)[
                               :10].reset_index()
players_xg: pd.DataFrame = shots.groupby(["player_name"])["our_xG"].sum().reset_index()

""" *** Data Vizualisation- Top 10 bar + p90 Scatterplot """

# Top 10 player xG plt

# Group by player_name and sum the xG values
# top_players: pd.DataFrame = shots.groupby(["player_name"])["our_xG"].sum().sort_values(ascending=False)[
#                            :10].reset_index()

# Plotting the horizontal bar chart
# fig, ax = plt.subplots(figsize=(10, 6))
# bars = ax.barh(top_players["player_name"], top_players["our_xG"], color='blue')
# Hide spines and x-axis
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.xaxis.set_visible(False)
# ax.bar_label(bars, fmt='%.1fxG', label_type='edge', color='black', fontsize=8)
# Set title and adjust layout
# plt.title('Top 10 Players by Total xG')
# plt.tight_layout()

# Show the plot
# plt.show()

# Scatter viz
# Read in general player stats from euro 2020 to obtain minutes played- Note due to the lack of minutes played data
# in the statsbomb free data we have parsed minutes played from the fbref website. We are unable to merge the minutes
# played data for all players due to the differences in names used between fbref and stasbomb e.g. Diogo Jota vs
# Diogo José Teixeira da Silva. As such we have a visualisation for 60 goalscorers at Euro 2020

# Obtain data
file_path = 'euro2020.csv'
m_played: pd.DataFrame = pd.read_csv(file_path)
m_played = m_played.rename(columns={"Player_name": 'player_name'})

# Create a dataframe containing sum of all players open play goals
players_g: pd.DataFrame = shots[shots["outcome_name"] == "Goal"].groupby("player_name").size().reset_index(
    name='total_goals')

# Merge datasets
players_total: pd.DataFrame = pd.merge(players_g, m_played, on='player_name', how='inner')
players_total = pd.merge(players_total, players_xg, on='player_name', how='inner')

# Apply the condition and calculate per 90 stats
players_total.loc[players_total['Player_Minutes'] > 90, "Goals P90"] = (
        (players_total["total_goals"] / players_total['Player_Minutes']) * 90)

players_total.loc[players_total['Player_Minutes'] > 90, "xG P90"] = (
        (players_total["our_xG"] / players_total['Player_Minutes']) * 90)

players_total['Difference'] = (players_total["Goals P90"] - players_total["xG P90"])

# Create X and Y variable to equate to goals and xG
x: pd.Series = players_total['xG P90']
y: pd.Series = players_total['Goals P90']
diff: pd.Series = players_total['Difference']

# Get surnames only
players_total['surname'] = players_total['player_name'].apply(lambda x: x.split()[-1])

# Scatter viz
plt.style.use('seaborn')

scatter = plt.scatter(x, y, s=100, c=diff, cmap='YlGn',
                      edgecolors='black', linewidths=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('xG Over performance')

# Set axis increments
new_xticks = np.arange(0, 1, 0.2)
plt.xticks(new_xticks)
new_yticks = np.arange(0, 1.6, 0.2)
plt.yticks(new_yticks)
# Set axis labels and plot title
plt.xlabel('Expected Goals (xG) p90')
plt.ylabel('Actual Goals Score p90')
plt.title('Euro 2020: xG p90 vs Actual Goals Scored p90')

# Adding data labels
# Identify the top 10 players based on the "Difference" column
top_10_players: pd.DataFrame = players_total.nlargest(10, 'Difference')

# Annotate the top 10 players on the scatterplot
for i, player in top_10_players.iterrows():
    plt.annotate(player['player_name'], (player['xG P90'], player['Goals P90']),
                 textcoords="offset points", xytext=(4, 4), ha='left')

# Show the plot
plt.show()

# todo add poetry
# todo understand .py test
