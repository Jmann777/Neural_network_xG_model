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
from Source import model_var_setup as mvs, statsbomb_jm as sb, model_training as mt

# Setting random seeds in order to allow for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

""" ***** Data preparation *****"""

shots, tracking_df = mt.data_prep_model()
model_vars = mt.model_vars()

""" ***** Model prediction ***** """


def model_prediction() -> pd.DataFrame:
    """ Predicts xG using the scaler and model trained in model_training.py

        Args:
    - model_vars (np.ndarray): Variables for model prediction

    Returns:
    - shots (pd.Dataframe): Dataframe with predicted xG assigned
    """
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
    return shots


shots = model_prediction()
# Group players and save results
top_players_xg: pd.DataFrame = shots.groupby(["player_name"])["our_xg"].sum().sort_values(ascending=False)[
                               :10].reset_index()
players_xg: pd.DataFrame = shots.groupby(["player_name", "player_id"])["our_xg"].sum().reset_index()
players_xg.to_csv('euros_predicted_goals.csv', index=False)
