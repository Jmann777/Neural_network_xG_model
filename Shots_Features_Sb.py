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
    # get id of the shot to search for tracking data using this index
    test_shot_id = test_shot["id"]
    # check goalkeeper position
    gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
    # calculate distance from event to goalkeeper position
    dist = np.sqrt((test_shot["x"] - gk_pos["x"])**2 + (test_shot["y"] - gk_pos["y"])**2)
    return dist.iloc[0]



