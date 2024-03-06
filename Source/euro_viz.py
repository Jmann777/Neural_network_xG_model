""" The following script creates two data visualisations based on the results within euros_results.py.

The visualisations are as follows:  - Bar Chart of the top ten highest xG players at the euros
                                    - Scatter plot of xG per 90 vs actual goals per 90"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from mplsoccer import VerticalPitch
from highlight_text import fig_text
from Source import  euros_results, jmann_viz_setup as jvs

""" ***** Data preparation ***** """

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

dmfont: dict[str, str] = {'family': 'DM Sans'}

# Top 10 xG Barchart
fig = plt.figure(figsize=(6, 2.5), dpi=200, facecolor="#bebfc4")
ax = plt.subplot(111, facecolor="#bebfc4")

width = 0.5

# Add spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add grid and axis labels
ax.grid(True, color="lightgrey", ls=":")


width = 0.5
height = top_players["our_xg"]


bars = ax.barh(
    top_players["player_name"],
    top_players["our_xg"],
    ec="black",
    lw=.75,
    color='#138015',
    zorder=3,
)

ax.bar_label(bars, **dmfont, fmt='%.1fxG',
             label_type='edge', padding=0.5,
             fontsize=5, color='#138015', fontweight='bold')
ax.tick_params(labelsize=8)

fig_text(
    x=0.23, y=0.95,
    s="Top 10 player with the highest expected goals for Euro 2020 based on a neural network model",
    **dmfont,
    color="black",
    size=10
)

fig_text(
    x=0.23, y=0.9,
    s="Viz by Josh Mann",
    **dmfont,
    color="#565756",
    size=8
)

# Add logo
jvs.watermark(ax, 10, 6)

plt.show()

# Scatter plot of xG p90 vs actual goals p90
fig = plt.figure(figsize=(8, 6), facecolor='#bebfc4')
ax = plt.subplot(111, facecolor="#bebfc4")
scatter = ax.scatter(x, y, s=100, c=diff, cmap='YlGn',
                     edgecolors='black', linewidths=1, alpha=0.75)
# Customisation
plt.style.use('seaborn')
cbar = fig.colorbar(scatter, ax=ax, label='xG Overperformance')
cbar.set_ticks(np.arange(0, 0.78, 0.25))
ax.set_xticks(np.arange(0, 1, 0.2))
ax.set_yticks(np.arange(0, 1.6, 0.2))
ax.set_xlabel('Expected Goals (xG) p90')
ax.set_ylabel('Actual Goals Score p90')
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', alpha=0.7)
# Add logo
jvs.watermark(ax, 8, 6)
# Annotate the top 10 players on the scatterplot
for i, player in top_10_players.iterrows():
    plt.annotate(player['player_name'], (player['xG P90'], player['Goals P90']),
                 textcoords="offset points", xytext=(4, 4), ha='left')

fig_text(
    x=0.15, y=0.95,
    s="Euro 2020: xG p90 vs Actual Goals Scored p90",
    **dmfont,
    color="black",
    size=20
)

fig_text(
    x=0.15, y=0.9,
    s="Viz by Josh Mann",
    **dmfont,
    color="#565756",
    size=16
)
# Show plot
plt.show()

#todo shotmap of top 10 highest xG players - https://www.sonofacorner.com/shot-maps-a-matplotlib-tutorial/
# todo can I put these into functions?
# Top 10 xG shot maps
# Draw pitch
fig = plt.figure(figsize=(4, 4), dpi=100)
ax = plt.subplot(111)
vertical_pitch = VerticalPitch(
    pitch_type="uefa",
    half=True,
    axis=True,
    label=True,
    tick=True,
    goal_type='box'
)


#def top10_xg_mapplot(ax, grids, player_name, data=top_players):
    '''
    This plots our shot heat map based on the grids defined
    by the soc_pitch_divisions function.

    Parameters:
    - ax (obj): a matplotlib Axes object.
    - grids (bool): whether or not to plot the grids.
    - player_name (int): the name of the player plotted
    - data (pd.DataFrame): the data

    Returns:
    - ax (obj): final matplotlib axes object
    '''

    df = data.copy()
    df = data[data["player_name"] == player_name]
    total_xg = df["our_xg"].sum()

    df = (
        df
        .assign(xGOT_share=lambda x: x.xGOT / total_xg)
    )
    df = (
        df
        .assign(xGOT_scaled=lambda x: x.xGOT_share / x.xGOT_share.max())
    )

    soc_pitch_divisions(ax, grids=grids)

    counter = 0
    for X, Y in zip(df["bins_x"], df["bins_y"]):
        ax.fill_between(
            x=[X.left, X.right],
            y1=Y.left,
            y2=Y.right,
            color="#495371",
            alpha=df["xGOT_scaled"].iloc[counter],
            zorder=-1,
            lw=0
        )

        if df['xGOT_share'].iloc[counter] > .02:
            text_ = ax.annotate(
                xy=(X.right - (X.right - X.left) / 2, Y.right - (Y.right - Y.left) / 2),
                text=f"{df['xGOT_share'].iloc[counter]:.0%}",
                ha="center",
                va="center",
                color="black",
                size=6.5,
                weight="bold",
                zorder=3
            )

            text_.set_path_effects(
                [path_effects.Stroke(linewidth=1.5, foreground="white"), path_effects.Normal()]
            )

        counter += 1

    return ax

#todo add saved csvs