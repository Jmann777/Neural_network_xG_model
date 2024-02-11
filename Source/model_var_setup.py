"""The following file creates the features used as independent variables within the neural network trained in
model_training.py. It consists of a basic distance and angle based xG calculation combined with goalkeeper location and
close player location
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def default_model_vars(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Obtain goals (dependent variable), shot location(IV), and calculate shot angle and distance (IVs).

    Parameters:
    - shots (pd.DataFrame): Dataframe containing open play shots

    Returns:
    - model_vars (pd.Dataframe): Dataframe containing model variables including goals(DV) location, angle, distance(IVs)
    """
    model_vars: pd.DataFrame = shots[["id", "index", "x", "y"]]
    # Get the dependent variable (goals)
    model_vars["goal"] = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    model_vars["goal_smf"] = model_vars['goal'].astype(object)
    # ball location (x)
    model_vars['x_ball'] = model_vars.x
    # Calculating angle and distance
    model_vars["x"] = model_vars.x.apply(lambda cell: 105 - cell)
    model_vars["c"] = model_vars.y.apply(lambda cell: abs(34 - cell))
    model_vars["angle"] = np.where(np.arctan(7.32 * model_vars["x"] / (
            model_vars["x"] ** 2 + model_vars["c"] ** 2 - (7.32 / 2) ** 2)) >= 0, np.arctan(
        7.32 * model_vars["x"] / (model_vars["x"] ** 2 + model_vars["c"] ** 2 - (7.32 / 2) ** 2)), np.arctan(
        7.32 * model_vars["x"] / (model_vars["x"] ** 2 + model_vars["c"] ** 2 - (7.32 / 2) ** 2)) + np.pi) * 180 / np.pi
    model_vars["distance"] = np.sqrt(model_vars["x"] ** 2 + model_vars["c"] ** 2)
    return model_vars


def dist_shot_to_gk(test_shot: pd.DataFrame, track_df: pd.DataFrame) -> float:
    """
    Calculate the distance from a shot event to the goalkeeper position.

    Parameters:
    - test_shot (pd.DataFrame): DataFrame containing information about the shot event, including id and location
    - track_df (pd.DataFrame): DataFrame containing tracking data, including id, location, teammate, and position_name

    Returns:
    - float: Distance from the shot event to the goalkeeper position
    """
    test_shot_id: int = test_shot["id"]
    gk_pos: pd.DataFrame = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[
        track_df["position_name"] == "Goalkeeper"][["x", "y"]]
    # calculate distance from event to goalkeeper position
    distance: pd.Series = np.sqrt((test_shot["x"] - gk_pos["x"]) ** 2 + (test_shot["y"] - gk_pos["y"]) ** 2)
    return distance.iloc[0]


def y_dist_to_gk(test_shot: pd.DataFrame, track_df: pd.DataFrame) -> float:
    """
    Calculate the distance from a shot event to the goalkeeper position on the Y axis.

    Parameters:
    - test_shot (pd.DataFrame): DataFrame containing information about the shot event, including id and location
    - track_df (pd.DataFrame): DataFrame containing tracking data, including id, location, teammate, and position_name

    Returns:
    - float: Distance from the shot event to the goalkeeper position on the Y axis (length)
    """
    test_shot_id: int = test_shot["id"]
    gk_pos: pd.DataFrame = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[
        track_df["position_name"] == "Goalkeeper"][["y"]]
    # calculate distance from event to goalkeeper position in y-axis
    y_distance: pd.Series = abs(test_shot["y"] - gk_pos["y"])
    return y_distance.iloc[0]


def three_meters_away(test_shot: pd.DataFrame, track_df: pd.DataFrame) -> int:
    """
    Count the number of opposition players within 3 meters of a shot event.

    Parameters:
    - test_shot (pd.DataFrame): DataFrame containing information about the shot event, including id and location
    - track_df (pd.DataFrame): DataFrame containing tracking data, including id, location, teammate, and position_name

    Returns:
    - int: Number of opposition players within 3 meters of the shot event
    """
    test_shot_id: int = test_shot["id"]
    player_position: pd.DataFrame = track_df.loc[(track_df["id"] == test_shot_id) & (track_df["teammate"] == False)][
        ["x", "y"]]
    # calculate opposition player distance to the ball
    opp_distance: pd.Series = np.sqrt(
        (test_shot["x"] - player_position["x"]) ** 2 + (test_shot["y"] - player_position["y"]) ** 2)
    return len(opp_distance[opp_distance < 3])


def players_in_triangle(test_shot: pd.DataFrame, track_df: pd.DataFrame) -> int:
    """
    Count the number of opposition players inside a triangle formed by specific coordinates around the ball.

    Parameters:
    - test_shot (pd.DataFrame): DataFrame containing information about the shot event, including id and location
    - track_df (pd.DataFrame): DataFrame containing tracking data, including id, location, teammate, and position_name

    Returns:
    - int: Number of opposition players inside the defined triangle.
    """
    # get id of the shot to search for tracking data using this index
    test_shot_id: int = test_shot["id"]
    # get all opposition's player location
    player_position: pd.DataFrame = track_df.loc[(track_df["id"] == test_shot_id) & (
            track_df["teammate"] == False)][["x", "y"]]
    # checking if point inside a triangle
    x1: float = 105
    y1: float = 34 - 7.32 / 2
    x2: float = 105
    y2: float = 34 + 7.32 / 2
    x3: float = test_shot["x"]
    y3: float = test_shot["y"]
    xp: pd.Series = player_position["x"]
    yp: pd.Series = player_position["y"]
    c1: pd.Series = (x2 - x1) * (yp - y1) - (y2 - y1) * (xp - x1)
    c2: pd.Series = (x3 - x2) * (yp - y2) - (y3 - y2) * (xp - x2)
    c3: pd.Series = (x1 - x3) * (yp - y3) - (y1 - y3) * (xp - x3)
    return len(player_position.loc[((c1 < 0) & (c2 < 0) & (c3 < 0)) | ((c1 > 0) & (c2 > 0) & (c3 > 0))])


def gk_dist_to_goal(test_shot: pd.DataFrame, track_df: pd.DataFrame) -> float:
    """
    Calculate the distance from the goalkeeper's position to the goal.

    Parameters:
    - test_shot (pd.DataFrame): DataFrame containing information about the shot event, including id and location
    - track_df (pd.DataFrame): DataFrame containing tracking data, including id, location, teammate, and position_name

    Returns:
    - float: Distance from the Goalkeepers position to the goal
    """
    # get id of the shot to search for tracking data using this index
    test_shot_id: int = test_shot["id"]
    # get goalkeeper position
    gk_pos: pd.DataFrame = track_df.loc[(track_df["id"] == test_shot_id) & (
            track_df["teammate"] == False) & (track_df["position_name"] == "Goalkeeper")][["x", "y"]]
    # calculate their distance to goal
    gk_distance: pd.Series = np.sqrt((105 - gk_pos["x"]) ** 2 + (34 - gk_pos["y"]) ** 2)
    return gk_distance.iloc[0]


# Logistic regression to calculate xg
def params(df: pd.DataFrame) -> pd.Series:
    """
     Fit a Generalized Linear Model to the provided DataFrame and return the model parameters.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data for the GLM

    Returns:
    - pd.Series: Parameters of the fitted GLM model.
    """
    test_model = smf.glm(formula="goal_smf ~ angle + distance", data=df,
                         family=sm.families.Binomial()).fit()
    return test_model.params


def calculate_xg(sh: pd.Series, b: np.ndarray) -> float:
    """
    Calculate the expected goals (xG) based on model coefficients and input features.

    Parameters:
    - sh (pd.Series): Series containing input features
    - b (np.ndarray): Array containing model coefficients

    Returns:
    - float: Expected goals (xG) calculated using logistic function.
    """
    b_sum = b[0]
    for i, v in enumerate(["angle", "distance"]):
        b_sum = b_sum + b[i + 1] * sh[v]
    xg = 1 / (1 + np.exp(b_sum))
    return xg
