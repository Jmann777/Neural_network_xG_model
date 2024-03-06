""" The following script creates a player pizza plot for Kasper Dolberg"""

import pandas as pd
import statsbomb_jm as sb
import matplotlib.pyplot as plt

from mplsoccer import PyPizza, add_image
from scipy import stats
from typing import Tuple, List

""" **** Data preparation **** """

competition_id: int = 55
season_id: int = 43

file_path_xg: str = 'euros_predicted_goals.csv'
modelled_xg: pd.DataFrame = pd.read_csv(file_path_xg)
events: pd.DataFrame = sb.events_season(competition_id, season_id)
passes: pd.DataFrame = events.loc[events["type_name"] == "Pass"]
file_path_shots: str = 'euros_shot_xg.csv'
shots: pd.DataFrame = pd.read_csv(file_path_shots)
events_xg_merge: pd.DataFrame = events.merge(shots[['id', 'our_xg']], how='left', on='id')
# Obtain minutes played data
file_path_min: str = '../euro2020.csv'
m_played: pd.DataFrame = pd.read_csv(file_path_min)
m_played = m_played.rename(columns={"Player_name": 'player_name'})


def npshots_npg_xg(shot_data: pd.DataFrame) -> pd.DataFrame:
    """ Obtains and adjusts the number shots, non-penalty goals, and the xG of all players per 90

    Parameters:
    - shot_data (pd.DataFrame): Dataframe containing shot data from statsbomb

    Returns:
    - shot_data (pd.DataFrame): Dataframe containing shot related data for all players. Specifically, number of shots,
    non-penalty goals, and xG per 90.
    """
    # Assuming 'shots' DataFrame contains 'player_id' and 'player_name' columns
    shots_player: pd.DataFrame = shots.groupby(["player_id", "player_name"]).size().reset_index(name='shots')
    shot_per90: pd.DataFrame = shots_player.merge(m_played, on="player_name")
    shot_per90.loc[shot_per90['Player_Minutes'] > 90, "shot_p90"] = (
            (shot_per90["shots"] / shot_per90['Player_Minutes']) * 90)

    goals: pd.Series = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    shots["non_pen_goals"] = goals
    npg_player: pd.DataFrame = shots[["player_id", "player_name", "non_pen_goals"]]
    npg_player = npg_player.groupby(["player_id", "player_name"])["non_pen_goals"].sum().reset_index()
    npg_per90: pd.DataFrame = npg_player.merge(m_played, on="player_name")
    npg_per90.loc[npg_per90['Player_Minutes'] > 90, "npg_p90"] = (
            (npg_per90["non_pen_goals"] / npg_per90['Player_Minutes']) * 90)

    xg_player: pd.DataFrame = modelled_xg
    xg_per90: pd.DataFrame = xg_player.merge(m_played, on="player_name")
    xg_per90.loc[xg_per90['Player_Minutes'] > 90, "xg_p90"] = (
            (xg_per90["our_xg"] / xg_per90['Player_Minutes']) * 90)
    xg_per90 = xg_per90.sort_values(by='player_id').reset_index(drop=True)

    shot_stats: pd.DataFrame = pd.concat([shot_per90,
                                          npg_per90.drop(columns=['player_name', 'Player_Minutes', 'player_id']),
                                          xg_per90.drop(columns=['player_name', 'Player_Minutes', 'player_id'])],
                                         axis=1)

    return shot_stats


def assists_kp_kpxg(passes: pd.DataFrame, events_xg: pd.DataFrame) -> pd.DataFrame:
    """ Obtains and adjusts the number assists, key passes, and the key pass xG of all players per 90.
    Key passes are defined as passes that resulted in a shot

    Parameters:
    - passes (pd.DataFrame): Dataframe containing pass data from statsbomb
    - events_xg (pd.DataFrame): Dataframe containing all events and xG values

    Returns:
    - pass_data (pd.DataFrame): Dataframe containing pass related data for all players. Specifically, number of assists,
    key passes, and key pass xG per 90.
    """
    assists: pd.Series = passes.pass_goal_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
    passes['assists'] = assists
    assist_player: pd.DataFrame = passes[["player_id", "player_name", "assists"]]
    assist_player = assist_player.groupby(["player_id", "player_name"])["assists"].sum().reset_index()
    assist_per90: pd.DataFrame = assist_player.merge(m_played, on="player_name")
    assist_per90.loc[assist_per90['Player_Minutes'] > 90, "assists_p90"] = (
            (assist_per90["assists"] / assist_per90['Player_Minutes']) * 90)
    # assist_per90["assists_p90"] = assist_per90["assists"] / assist_per90["Player_Minutes"] * 90

    key_passes: pd.Series = passes.pass_shot_assist.fillna(0).apply(lambda cell: 0 if cell == 0 else 1)
    passes['key_passes'] = key_passes
    key_passes_player: pd.DataFrame = passes[["player_id", "player_name", "key_passes"]]
    key_passes_player = key_passes_player.groupby(["player_id", "player_name"])["key_passes"].sum().reset_index()
    key_passes_per90: pd.DataFrame = key_passes_player.merge(m_played, on="player_name")
    key_passes_per90.loc[key_passes_per90['Player_Minutes'] > 90, "key_pass_p90"] = (
            (key_passes_per90["key_passes"] / key_passes_per90['Player_Minutes']) * 90)

    events_xg['key_passes'] = key_passes
    key_pass_events: pd.DataFrame = events_xg.loc[
        (events_xg["type_name"].isin(["Pass", "Shot"])) & (
                (events_xg["key_passes"] == 1) | (events_xg["type_name"] == "Shot"))]
    key_pass_events = key_pass_events.sort_values(by="possession")
    num_possession: int = max(key_pass_events["possession"].unique())
    for i in range(num_possession + 1):
        possession_chain: pd.DataFrame = key_pass_events.loc[key_pass_events["possession"] == i].sort_values(by="index")
        if len(possession_chain) > 0:
            if possession_chain.iloc[-1]["type_name"] == "Shot":
                xg: float = possession_chain.iloc[-1]["our_xg"]
                key_pass_events.loc[key_pass_events["possession"] == i, 'our_xg'] = xg
    key_pass_events = key_pass_events.loc[key_pass_events["key_passes"] == 1]
    key_pass_xg_player: pd.DataFrame = key_pass_events.groupby("player_name")["our_xg"].sum().reset_index()
    key_pass_xg_player.rename(columns={"our_xg": "key_pass_xg"}, inplace=True)
    key_pass_xg_player = key_pass_xg_player.merge(m_played, on="player_name")
    key_pass_xg_player.loc[key_pass_xg_player['Player_Minutes'] > 90, "key_pass_xg_p90"] = (
            (key_pass_xg_player["key_pass_xg"] / key_pass_xg_player['Player_Minutes']) * 90)

    pass_stats: pd.DataFrame = pd.concat([assist_per90,
                                          key_passes_per90.drop(columns=['player_name', 'Player_Minutes', 'player_id']),
                                          key_pass_xg_player.drop(columns=['player_name', 'Player_Minutes'])], axis=1)

    return pass_stats


shot_stats: pd.DataFrame = npshots_npg_xg(shots)
pass_stat: pd.DataFrame = assists_kp_kpxg(passes, events_xg_merge)

combined_stats: pd.DataFrame = pd.concat([shot_stats,
                                          pass_stat.drop(columns=['player_name', 'Player_Minutes'])], axis=1)

""" ***** Data visualisations *****"""


def pizza_data_setup(combined_stats: pd.DataFrame) -> Tuple[List[int], List[str], List[str], List[str]]:
    # Dolberg Viz
    player: pd.DataFrame = combined_stats.loc[combined_stats["player_name"] == "Kasper Dolberg"]
    player = player[['non_pen_goals', 'npg_p90', 'our_xg', 'xg_p90', 'shots', 'shot_p90',
                     'assists', 'assists_p90', 'key_passes', 'key_pass_p90', 'key_pass_xg', 'key_pass_xg_p90']]
    player_viz: List[str] = player.columns[:]
    player_val: List[float] = [round(player[column].iloc[0], 2) for column in player_viz]
    combined_viz = combined_stats.fillna(0, inplace=True)
    player_percentiles: List[int] = [int(stats.percentileofscore(
        combined_stats[column], player[column].iloc[0])) for column in player_viz]

    att_names: List[str] = ["Non-Penalty Goals", "Non-Penalty Goals p90", "xG", "xG p90", "Shots", "Shots p90",
                            "Assists", "Assists p90", "key passes", "key passes p90", "key pass xg", "key pass xg p90"]
    slice_colors: List[str] = ["#138015"] * 6 + ["#FFD449"] * 6
    text_colors: List[str] = ["#000000"] * 12
    return player_percentiles, att_names, slice_colors, text_colors


def pizza_plot(params: List[str], data: List[float], s_colors: List[str], t_colors: List[str]):
    baker = PyPizza(
        params=params,
        background_color="#bebfc4",
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=20
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        data,
        figsize=(8, 8.5),
        color_blank_space="same",
        slice_colors=s_colors,
        value_colors=t_colors,
        value_bck_colors=s_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(
            edgecolor="#000000", zorder=2, linewidth=1
        ),
        kwargs_params=dict(
            color="#000000", fontsize=11, va="center"
        ),
        kwargs_values=dict(
            color="#000000", fontsize=11, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )
    )

    # add title
    fig.text(
        0.5, 0.975, "Kasper Dolberg - Denmark", size=16,
        ha="center", weight='bold', color="#000000"
    )

    # add subtitle
    fig.text(
        0.5, 0.955,
        "Percentile Rank vs All Players | European Championship 2020",
        size=13,
        ha="center", color="#000000"
    )

    sub = "Viz by Josh Mann"

    fig.text(
        0.99, 0.02, f"{sub}", size=9, color="#000000",
        ha="right"
    )

    # add text
    fig.text(
        0.409, 0.9225, "Attacking- Shooting", size=13,
        color="#000000"
    )

    fig.text(
        0.56, 0.9225, "Attacking- Playmaking", size=13,
        color="#000000"
    )

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.38, 0.9225), 0.025, 0.021, fill=True, color="#138015",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.532, 0.9225), 0.025, 0.021, fill=True, color="#FFD449",
            transform=fig.transFigure, figure=fig
        ),
    ])
    # add logo
    logo = plt.imread('jmann-logo.png')
    ax_image = add_image(
        logo, fig, left=0.430, bottom=0.419, width=0.165, height=0.152)


player_percentiles, att_names, slice_colors, text_colors = pizza_data_setup(combined_stats)

pizza_plot(att_names, player_percentiles, slice_colors, text_colors)

plt.show()

# todo extra plot filter to strikers in Euros?
