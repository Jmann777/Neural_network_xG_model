import numpy as np


def default_model_vars(shots):
    # Take the important variables from the shot dataframe
    model_vars = shots[["id", "index", "x", "y"]]
    # Get the dependent variable (goals)
    model_vars["goal"] = shots.outcome_name.apply(lambda cell: 1 if cell == 'Goal' else 0)
    # Change dependent variable to object for basic xG modelling (often done for categorical or binary variables)
    model_vars["goal_smf"] = model_vars['goal'].astype(object)
    # ball location (x)
    model_vars['x0'] = model_vars.x
    # x to calculate angle and distance
    model_vars["x"] = model_vars.x.apply(lambda cell: 105 - cell)
    # c to calculate angle and distance between ball and the goal as in Lesson 2
    model_vars["c"] = model_vars.y.apply(lambda cell: abs(34 - cell))
    # Calculating angle and distance as in Lesson 2
    model_vars["angle"] = np.where(np.arctan(7.32 * model_vars["x"] / (
            model_vars["x"] ** 2 + model_vars["c"] ** 2 - (7.32 / 2) ** 2)) >= 0, np.arctan(
        7.32 * model_vars["x"] / (model_vars["x"] ** 2 + model_vars["c"] ** 2 - (7.32 / 2) ** 2)), np.arctan(
        7.32 * model_vars["x"] / (model_vars["x"] ** 2 + model_vars["c"] ** 2 - (7.32 / 2) ** 2)) + np.pi) * 180 / np.pi
    model_vars["distance"] = np.sqrt(model_vars["x"] ** 2 + model_vars["c"] ** 2)
    return model_vars


def dist_to_gk(test_shot, track_df):
    # get id of the shot to search for tracking data using this index- test shot in local to this function
    test_shot_id = test_shot["id"]
    # check goalkeeper position
    gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[
        track_df["position_name"] == "Goalkeeper"][["x", "y"]]
    # calculate distance from event to goalkeeper position
    dist = np.sqrt((test_shot["x"] - gk_pos["x"]) ** 2 + (test_shot["y"] - gk_pos["y"]) ** 2)
    return dist.iloc[0]


# ball goalkeeper y axis
def y_to_gk(test_shot, track_df):
    # get id of the shot to search for tracking data using this index
    test_shot_id = test_shot["id"]
    # calculate distance from event to goalkeeper position
    gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[
        track_df["position_name"] == "Goalkeeper"][["y"]]
    # calculate distance from event to goalkeeper position in y axis
    dist = abs(test_shot["y"] - gk_pos["y"])
    return dist.iloc[0]

# number of players less than 3 meters away from the ball
def three_meters_away(test_shot, track_df):
    # get id of the shot to search for tracking data using this index
    test_shot_id = test_shot["id"]
    # get all opposition's player location
    player_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
    # calculate their distance to the ball
    dist = np.sqrt((test_shot["x"] - player_position["x"]) ** 2 + (test_shot["y"] - player_position["y"]) ** 2)
    # return how many are closer to the ball than 3 meters
    return len(dist[dist < 3])

# number of players inside a triangle
def players_in_triangle(test_shot, track_df):
    # get id of the shot to search for tracking data using this index
    test_shot_id = test_shot["id"]
    # get all opposition's player location
    player_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
    # checking if point inside a triangle
    x1 = 105
    y1 = 34 - 7.32 / 2
    x2 = 105
    y2 = 34 + 7.32 / 2
    x3 = test_shot["x"]
    y3 = test_shot["y"]
    xp = player_position["x"]
    yp = player_position["y"]
    c1 = (x2 - x1) * (yp - y1) - (y2 - y1) * (xp - x1)
    c2 = (x3 - x2) * (yp - y2) - (y3 - y2) * (xp - x2)
    c3 = (x1 - x3) * (yp - y3) - (y1 - y3) * (xp - x3)
    # get number of players inside a triangle
    return len(player_position.loc[((c1 < 0) & (c2 < 0) & (c3 < 0)) | ((c1 > 0) & (c2 > 0) & (c3 > 0))])

# goalkeeper distance to goal
def gk_dist_to_goal(test_shot, track_df):
    # get id of the shot to search for tracking data using this index
    test_shot_id = test_shot["id"]
    # get goalkeeper position
    gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[
        track_df["position_name"] == "Goalkeeper"][["x", "y"]]
    # calculate their distance to goal
    dist = np.sqrt((105 - gk_pos["x"]) ** 2 + (34 - gk_pos["y"]) ** 2)
    return dist.iloc[0]