""" The following script creates two data visualisations based on the results within euros_results.py.

The visualisations are as follows:  - Bar Chart of the top ten highest xG players at the euros
                                    - Scatter plot of xG per 90 vs actual goals per 90"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from highlight_text import fig_text
from Source import euros_results, jmann_viz_setup as jvs


def data_prep_viz():
    """Prepares the variables needed to create the bar chart and scatter plot visualisation.

    Parameters:
    - player_mintues (pd.DataFrame): Dataframe containing the minutes played data for each player at Euro 2020

    Returns:
    - top_10_players (pd.DataFrame): Dataframe containing the top 10 players in a certain statistic.
    In this specific case, top 10 players with the highest xG at Euro 2020.
    - x (pd.Series): Series containing xG per 90 values for each player at Euro 2020 (min of 90 mins played to qualify)
    - y (pd.Series): Series containing the number of actual goals scored per player at Euro 2020 (min of 90 mins played)
    """
    top_players: pd.DataFrame = euros_results.shots.groupby(
        ["player_name"])["our_xg"].sum().sort_values(ascending=False)[:10].reset_index()
    file_path = '../euro2020.csv'
    m_played: pd.DataFrame = pd.read_csv(file_path)
    m_played = m_played.rename(columns={"Player_name": 'player_name'})

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
    return top_players, top_10_players, x, y, diff


top_players, top_10_players, x, y, diff = data_prep_viz()

# State font required for visualisations
dmfont: dict[str, str] = {'family': 'DM Sans'}

""" ***** Data visualisations *****"""


def top_10_barchart(top_10: pd.DataFrame, font: dict):
    """ Creates a barchart visualisation of the top 10 players within a certain statistic. Specific to this project,
    this function creates a barchart of the top 10 players with the highest xG at Euro 2020.

    Parameters:
    - top_10 (pd.DataFrame): Dataframe containing the top 10 players in a certain statistic.
    In this specific case, top 10 players with the highest xG at Euro 2020.
    - font (dict): Dictionary strings related to the font use for text within the barchart
    """
    fig = plt.figure(figsize=(6, 2.5), dpi=200, facecolor="#bebfc4")
    ax = plt.subplot(111, facecolor="#bebfc4")
    width = 0.5

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, color="lightgrey", ls=":")

    # width = 0.5 #todo check if this is needed
    height = top_10["our_xg"]

    bars = ax.barh(
        top_10["player_name"],
        top_10["our_xg"],
        ec="black",
        lw=.75,
        color='#138015',
        zorder=3,
    )

    ax.bar_label(bars, **font, fmt='%.1fxG',
                 label_type='edge', padding=0.5,
                 fontsize=5, color='#138015', fontweight='bold')
    ax.tick_params(labelsize=8)

    fig_text(
        x=0.23, y=0.95,
        s="Top 10 player with the highest expected goals for Euro 2020 based on a neural network model",
        **font,
        color="black",
        size=10
    )

    fig_text(
        x=0.23, y=0.9,
        s="Viz by Josh Mann",
        **font,
        color="#565756",
        size=8
    )

    # Add logo
    jvs.watermark(ax, 10, 6)


top_10_barchart(top_players, dmfont)
plt.show()


def player_scatter(x: pd.Series, y: pd.Series, diff: pd.Series, font: dict):
    """ Creates a scatter plot of player statistics. Specific to this project, the scatter is based on
    actual goals scored per 90 minutes played vs xG per 90 minutes. This is to identify players who overperform
    their xG.

    Parameters:
    - x (pd.Series): Series containing xG per 90 values for each player at Euro 2020 (min of 90 mins played to qualify).
    - y (pd.Series): Series containing the number of actual goals scored per player at Euro 2020 (min of 90 mins played)
    - font (dict): Dictionary strings related to the font use for text within the barchart
    """
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
        **font,
        color="black",
        size=20
    )

    fig_text(
        x=0.15, y=0.9,
        s="Viz by Josh Mann",
        **font,
        color="#565756",
        size=16
    )


player_scatter(x, y, diff, dmfont)
plt.show()
