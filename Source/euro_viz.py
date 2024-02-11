""" The following script creates four data visualisations based on the results within euros_results.py.

The visualisations are as follows:  - Bar Chart of the top ten highest xG players at the euros
                                    - Shot maps of the top ten highest xG players at the euros
                                    - Scatter plot of xG per 90 vs actual goals per 90
                                    - Player radar of Kasper Dolberg"""

import euros_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects

from mplsoccer import VerticalPitch

""" ***** Data preparation for visualisation ***** """
# Bar chart and shot data prep
top_players: pd.DataFrame = euros_results.shots.groupby(
    ["player_name"])["our_xg"].sum().sort_values(ascending=False)[:10].reset_index()

# Scatter plot data prep
# Obtain minutes played data
file_path = '../euro2020.csv'
m_played: pd.DataFrame = pd.read_csv(file_path)
m_played = m_played.rename(columns={"Player_name": 'player_name'})
# Create a dataframe containing sum of all players open play goals
players_g: pd.DataFrame = euros_results.shots[
    euros_results.shots["outcome_name"] == "Goal"].groupby("player_name").size().reset_index(
    name='total_goals')
# Merge datasets
players_total: pd.DataFrame = pd.merge(players_g, m_played, on='player_name', how='inner')
players_total = pd.merge(players_total, euros_results.players_xg, on='player_name', how='inner')
# Apply the condition and calculate per 90 stats
players_total.loc[players_total['Player_Minutes'] > 90, "Goals P90"] = (
        (players_total["total_goals"] / players_total['Player_Minutes']) * 90)

players_total.loc[players_total['Player_Minutes'] > 90, "xG P90"] = (
        (players_total["our_xg"] / players_total['Player_Minutes']) * 90)

players_total['Difference'] = (players_total["Goals P90"] - players_total["xG P90"])

# Create X and Y variable to equate to goals and xG
x: pd.Series = players_total['xG P90']
y: pd.Series = players_total['Goals P90']
diff: pd.Series = players_total['Difference']

# Get surnames only
players_total['surname'] = players_total['player_name'].apply(lambda x: x.split()[-1])

# Identify the top 10 players based on the "Difference" column
top_10_players: pd.DataFrame = players_total.nlargest(10, 'Difference')

""" ***** Data visualisations *****"""

# Top 10 xG Barchart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_players["player_name"], top_players["our_xg"], color='Green')
# Customisation
plt.style.use('seaborn')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_visible(False)
ax.bar_label(bars, fmt='%.1fxG', label_type='edge', color='black', fontsize=8)
plt.title('Top 10 Players by Total xG', fontname='Times New Roman', size=20)
plt.tight_layout()

# Show the plot
plt.show()

# Top 10 xG shot maps
#fig = plt.figure(figsize=(4, 4), dpi=100)
#ax = plt.subplot(111)

#pitch = VerticalPitch()
#pitch.draw(ax=ax)

# Scatter plot of xG p90 vs actual goals p90
fig, ax = plt.subplots(figsize=(8, 6))  # todo Set custom figure size
scatter = ax.scatter(x, y, s=100, c=diff, cmap='YlGn',
                     edgecolors='black', linewidths=1, alpha=0.75)
# Customisation
plt.style.use('seaborn')
cbar = fig.colorbar(scatter, ax=ax, label='xG Overperformance')
cbar.set_ticks(np.arange(0, 0.6, 0.1))  # todo Set specific colorbar ticks
ax.set_xticks(np.arange(0, 1, 0.2))
ax.set_yticks(np.arange(0, 1.6, 0.2))
ax.set_xlabel('Expected Goals (xG) p90')
ax.set_ylabel('Actual Goals Score p90')
ax.set_title('Euro 2020: xG p90 vs Actual Goals Scored p90', fontweight='bold')
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.7)  # Add grid

# Annotate the top 10 players on the scatterplot
for i, player in top_10_players.iterrows():
    plt.annotate(player['player_name'], (player['xG P90'], player['Goals P90']),
                 textcoords="offset points", xytext=(4, 4), ha='left')

# Show plot
plt.show()

# todo make barchart green and create more viz- shotmap of top 10 highest xG players - https://www.sonofacorner.com/shot-maps-a-matplotlib-tutorial/
# todo make a consistent style- same colours, fonts,
# todo player radar of Kasper Dolberg?
# todo can I put these into functions?
