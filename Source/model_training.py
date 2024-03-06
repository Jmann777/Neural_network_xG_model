"""
The script will train the neural network model to predict xG on ISL to goal event and tracking data.
The model takes a combination of variables that include information taken from the Shots_features module.
After the model is run, visualisations of the model are created which include assessments of the model (AUC + ROC and
calibration curves).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Source import model, model_var_setup as mvs, statsbomb_jm as sb
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

''' ***** Data Prep *****'''

competition_id: int = 1238
season_id: int = 108
matches_df: pd.DataFrame = sb.matches(competition_id, season_id)
shots: pd.DataFrame = sb.shots_season(competition_id, season_id)
tracking_df: pd.DataFrame = sb.tracking_season(competition_id, season_id)

# Filtering data
shots = shots[shots["sub_type_name"] == "Open Play"]
gks_tracked: np.ndarray = tracking_df[tracking_df["teammate"] == False][
    tracking_df["position_name"] == "Goalkeeper"]['id'].unique()
shots = shots[shots["id"].isin(gks_tracked)]

''' ***** Model Prep- This includes the creation of features within the model taken from Shots_Features.py  *****'''

# Importing and creating model variables
model_vars: pd.DataFrame = mvs.default_model_vars(shots=shots)
b: pd.Series = mvs.params(model_vars)
model_vars["xg_basic"] = model_vars.apply(mvs.calculate_xg, b=b, axis=1)
model_vars["gk_distance"] = shots.apply(mvs.dist_shot_to_gk, track_df=tracking_df, axis=1)
model_vars["gk_distance_y_axis"] = shots.apply(mvs.y_dist_to_gk, track_df=tracking_df, axis=1)
model_vars["close_players"] = shots.apply(mvs.three_meters_away, track_df=tracking_df, axis=1)
model_vars["players_in_triangle"] = shots.apply(mvs.players_in_triangle, track_df=tracking_df, axis=1)
model_vars["gk_dist_to_goal"] = shots.apply(mvs.gk_dist_to_goal, track_df=tracking_df, axis=1)
model_vars["closer_to_goal"] = np.where(model_vars["gk_dist_to_goal"] > model_vars["distance"], 1, 0)
model_vars["header"] = shots.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)

# store dependent variable in a numpy array
y: np.ndarray = model_vars["goal"].values
# store independent variables in a numpy array
x: np.ndarray = model_vars[["x_ball", "closer_to_goal", "angle", "distance", "gk_distance", "gk_distance_y_axis",
                            "players_in_triangle", "close_players", "header", "xg_basic"]].values

''' ***** Model Training- Calling in the model from model.py  *****'''
# Calling in neural network from model.py
x_train, x_val, x_cal, y_train, y_val, y_cal = model.setup_model(x, y)
model, history = model.run_model(x_train, y_train, x_val, y_val)

fig, axs = plt.subplots(2, figsize=(10, 12))

# plot training history - accuracy
axs[0].plot(history['accuracy'], label='train')
axs[0].plot(history['val_accuracy'], label='validation')
axs[0].set_title("Accuracy at each epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()

# plot training history - loss function
axs[1].plot(history['loss'], label='train')
axs[1].plot(history['val_loss'], label='validation')
axs[1].legend()
axs[1].set_title("Loss at each epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MSE")
plt.show()

# Model Assessment using ROC(+AUC)
plt, axs = plt.subplots(2, figsize=(10, 12))
y_prediction: np.ndarray = model.predict(x_cal)
fpr, tpr, _ = roc_curve(y_cal, y_prediction)
auc: float = roc_auc_score(y_cal, y_prediction)
axs[0].plot(fpr, tpr, label="AUC = " + str(auc)[:4])
axs[0].plot([0, 1], [0, 1], color='black', ls='--')
axs[0].legend()
axs[0].set_ylabel('True Positive Rate')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_title('ROC curve')

# Model Assessment -Calibration curve- Actual probability vs predicted probability
prob_true, prob_prediction = calibration_curve(y_cal, y_prediction, n_bins=10)
axs[1].plot(prob_true, prob_prediction)
axs[1].plot([0, 1], [0, 1], color='black', ls='--')
axs[1].set_ylabel('Empirical Probability')
axs[1].set_xlabel('Predicted Probability')
axs[1].set_title("Calibration curve")
plt.show()


print("Brier score", brier_score_loss(y_cal, y_prediction))

#todo GIT! From our results we can see that our model is satisfactory, however it tends to assign more goals than in actuality
# when the probability of a goal is higher.
