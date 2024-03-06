""" The following seeks to test the model set up functions to ensure we are producing accurate results"""

import pytest
import pandas as pd
import numpy as np
import Source.model_var_setup


@pytest.fixture
def t_test_shot():
    return pd.DataFrame({
        "id": [0, 1],
        "index": [10, 20],
        "x": [50, 60],
        "y": [20, 30],
        "outcome_name": ["Goal", "Missed"]
    })


@pytest.fixture
def t_track_df():
    return pd.DataFrame({
        "id": [0, 1],
        "x": [45, 60],
        "y": [25, 32],
        "teammate": [False, False],
        "position_name": ["Goalkeeper", "Defender"]
    })


def test_gk_dist(t_test_shot: pd.DataFrame, t_track_df: pd.DataFrame):
    """Tests if the goalkeeper distance is calculated correctly"""
    gk_dist = Source.model_var_setup.gk_dist_to_goal(t_test_shot, t_track_df)
    expt_gk_dist = np.sqrt((105 - 45) ** 2 + (34 - 25) ** 2)  # Assuming goal at (105, 34) and GK at (45, 25)
    assert gk_dist == expt_gk_dist


def test_goal_variable_calculation(t_test_shot: pd.DataFrame):
    """Tests if the 'goal' variable is inputted correctly."""
    model_vars = Source.model_var_setup.default_model_vars(t_test_shot)
    expected_goals = [1, 0]
    assert list(model_vars["goal"]) == expected_goals


def test_dist_to_gk(t_test_shot: pd.DataFrame, t_track_df: pd.DataFrame):
    """Tests if shot distance to goalkeeper is calculated correctly"""
    dist_to_gk = Source.model_var_setup.dist_shot_to_gk(t_test_shot, t_track_df)
    expt_dist_to_gk = np.sqrt((50 - 45) ** 2 + (20 - 25) ** 2)
    assert dist_to_gk == expt_dist_to_gk


def test_y_to_gk(t_test_shot: pd.DataFrame, t_track_df: pd.DataFrame):
    """Tests that shot to goalkeeper distance along the y-axis is calculated correctly"""
    y_to_gk = Source.model_var_setup.y_dist_to_gk(t_test_shot, t_track_df)
    expt_y_to_gk = abs(20 - 25)
    assert y_to_gk == expt_y_to_gk


def test_three_meters_away(t_test_shot: pd.DataFrame, t_track_df: pd.DataFrame):
    """Tests the calculation of players within 3 meters of the ball"""
    three_meters_away = Source.model_var_setup.three_meters_away(t_test_shot, t_track_df)
    calc_three_meters_away = np.sqrt((60 - 60) ** 2 + (30 - 32) ** 2)
    expt_three_meters_away = len(calc_three_meters_away[calc_three_meters_away < 3])
    assert three_meters_away == expt_three_meters_away
