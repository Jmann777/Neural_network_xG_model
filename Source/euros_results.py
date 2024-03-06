"""
The following script uses the neural network created in model.py and trained in model_training.py to calculate
the xG for each player at Euro 2020. It then outputs the top 10 players with the highest xG
"""
import joblib
import os
import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd

from keras.models import load_model
from Source import model_var_setup as pdf, statsbomb_jm as sb

# Setting random seeds in order to allow for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

""" ***** Data preparation *****"""
competition_id: int = 55
season_id: int = 43
matches_df: pd.DataFrame = sb.matches(competition_id, season_id)
shots: pd.DataFrame = sb.shots_season(competition_id, season_id)
tracking_df: pd.DataFrame = sb.tracking_season(competition_id, season_id)

# Filtering data
shots = shots[shots["sub_type_name"] == "Open Play"]
gks_tracked: np.ndarray = tracking_df[
    tracking_df["teammate"] == False].loc[tracking_df["position_name"] == "Goalkeeper"]['id'].unique()
shots = shots[shots["id"].isin(gks_tracked)]

# Importing and creating model variables
model_vars: pd.DataFrame = pdf.default_model_vars(shots=shots)
b: pd.Series = pdf.params(model_vars)
model_vars["xg_basic"] = model_vars.apply(pdf.calculate_xg, b=b, axis=1)
model_vars["gk_distance"] = shots.apply(pdf.dist_shot_to_gk, track_df=tracking_df, axis=1)
model_vars["gk_distance_y_axis"] = shots.apply(pdf.y_dist_to_gk, track_df=tracking_df, axis=1)
model_vars["close_players"] = shots.apply(pdf.three_meters_away, track_df=tracking_df, axis=1)
model_vars["players_in_triangle"] = shots.apply(pdf.players_in_triangle, track_df=tracking_df, axis=1)
model_vars["gk_dist_to_goal"] = shots.apply(pdf.gk_dist_to_goal, track_df=tracking_df, axis=1)
model_vars["closer_to_goal"] = np.where(model_vars["gk_dist_to_goal"] > model_vars["distance"], 1, 0)
model_vars["header"] = shots.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)

""" ***** Model prediction ***** """
# load model and scaler
scaler = joblib.load('../fitted_scaler.joblib')
model = load_model('../best_model.h5')

# Store model vars into a matrix
x_unseen: np.ndarray = model_vars[["x_ball", "closer_to_goal", "angle", "distance",
                                   "gk_distance", "gk_distance_y_axis",
                                   "players_in_triangle", "close_players", "header", "xg_basic"]].values

# Scale independent variables
x_unseen: np.ndarray = scaler.transform(x_unseen)
# Make calculation on euros data
xgs_euro: np.ndarray = model.predict(x_unseen)

# Assign xG
shots["our_xg"] = xgs_euro
shots.to_csv('euros_shot_xg.csv', index=False)

# Group players and save results
top_players_xg: pd.DataFrame = shots.groupby(["player_name"])["our_xg"].sum().sort_values(ascending=False)[
                               :10].reset_index()
players_xg: pd.DataFrame = shots.groupby(["player_name", "player_id"])["our_xg"].sum().reset_index()
players_xg.to_csv('euros_predicted_goals.csv', index=False)
