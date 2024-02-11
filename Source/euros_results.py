"""
The following script using the neural network created in model.py and trained in model_training.py to calculate
the xG for each player at Euro 2020. It then outputs the top 10 players with the highest xG
"""
import joblib
import os
import numpy as np
import tensorflow as tf
import random as rn
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import load_model
from Source import model_var_setup as pdf, statsbomb as sb

# Setting random seeds in order to allow for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

""" ***** Data preparation for the model *****"""
competition_id: int = 55
season_id: int = 43
matches_df: pd.DataFrame = sb.matches(competition_id, season_id)
shots: pd.DataFrame = sb.shots_season(competition_id, season_id)
tracking_df: pd.DataFrame = sb.tracking_season(competition_id, season_id)

# Filtering out non open_play shots
shots = shots[shots["sub_type_name"] == "Open Play"]
# Filter out shots where goalkeeper was not tracked
gks_tracked: np.ndarray = tracking_df[
    tracking_df["teammate"] == False].loc[tracking_df["position_name"] == "Goalkeeper"]['id'].unique()
shots = shots[shots["id"].isin(gks_tracked)]

# Model variables
model_vars: pd.DataFrame = pdf.default_model_vars(shots=shots)
b: pd.Series = pdf.params(model_vars)
model_vars["xg_basic"] = model_vars.apply(pdf.calculate_xg, b=b, axis=1)
model_vars["gk_distance"] = shots.apply(pdf.dist_shot_to_gk, track_df=tracking_df, axis=1)
model_vars["gk_distance_y_axis"] = shots.apply(pdf.y_dist_to_gk, track_df=tracking_df, axis=1)
model_vars["close_players"] = shots.apply(pdf.three_meters_away, track_df=tracking_df, axis=1)
model_vars["players_in_triangle"] = shots.apply(pdf.players_in_triangle, track_df=tracking_df, axis=1)
model_vars["gk_dist_to_goal"] = shots.apply(pdf.gk_dist_to_goal, track_df=tracking_df, axis=1)
# create binary variable 1 if ball is closer to the goal than goalkeeper
model_vars["closer_to_goal"] = np.where(model_vars["gk_dist_to_goal"] > model_vars["distance"], 1, 0)
# create binary variable 1 if header
model_vars["header"] = shots.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)

""" ***** Loading and using the model to predict the top 10 players with highest xG according to the model in
Euro 2020. ***** """
# load scaler details
scaler = joblib.load('../fitted_scaler.joblib')
# load trained model
model = load_model('../best_model.h5')

# Store model vars into a matrix
x_unseen: np.ndarray = model_vars[["x_ball", "closer_to_goal", "angle", "distance",
                                   "gk_distance", "gk_distance_y_axis",
                                   "players_in_triangle", "close_players", "header", "xg_basic"]].values

# Scale independent variables
x_unseen: np.ndarray = scaler.transform(x_unseen)
# Make calculation on euros data
xgs_euro: np.ndarray = model.predict(x_unseen)
# Assign xG to the shots dataframe
shots["our_xg"] = xgs_euro
# Output the top 10 players with the highest xG
print(shots.groupby(["player_name"])["our_xg"].sum().sort_values(ascending=False)[:10].reset_index())
# Group by player_name and sum the xG values
top_players_xg: pd.DataFrame = shots.groupby(["player_name"])["our_xg"].sum().sort_values(ascending=False)[
                               :10].reset_index()
players_xg: pd.DataFrame = shots.groupby(["player_name"])["our_xg"].sum().reset_index()
