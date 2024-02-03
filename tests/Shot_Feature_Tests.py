import pytest
import pandas as pd
import numpy as np
import Source.Shots_Features_Sb

@pytest.fixture
def t_test_shot():
    return pd.DataFrame({
        "id": [1, 2],
        "x": [50, 60],
        "y": [20, 30],
        "outcome_name": ["Goal", "Missed"]
    })

@pytest.fixture
def t_track_df():
    return pd.DataFrame({
        "id": [1],
        "x": [45],
        "y": [25],
        "teammate": [False],
        "position_name": ["Goalkeeper"]
    })

def test_gk_dist (t_test_shot, t_track_df):
    result = Source.Shots_Features_Sb.gk_dist_to_goal(t_test_shot, t_track_df)
    expt_result = np.sqrt((105 - 45) ** 2 + (34 - 25) ** 2) # Assuming goal at (105, 34) and GK at (45, 25)
    assert result == expt_result
