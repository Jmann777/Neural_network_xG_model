"""
The script will train the neural network model to predict xG on 2015-16 Premier League goal event and tracking data.
The model takes a combination of variables that include information taken from the Shots_features module.
After the model is run, visualisations of the model are created which include assessments of the model (AUC + ROC and
calibration curves).
"""

# Importing modules
import Model
import numpy as np
import matplotlib.pyplot as plt
import Statsbomb as Sb
import Shots_Features_Sb as pdf

# Import machine learning libraries
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

''' ***** Data Prep *****'''
# Opening data
cid = 1238
sid = 108
df_match = Sb.sb_matches(cid, sid)
shots = Sb.sb_shots_season(cid, sid)
track_df = Sb.sb_tracking_season(cid, sid)

# Filtering out non open_play shots
shots = shots[shots["sub_type_name"] == "Open Play"]
# Filter out shots where goalkeeper was not tracked
gks_tracked = track_df[track_df["teammate"] == False][track_df["position_name"] == "Goalkeeper"]['id'].unique()
shots = shots[shots["id"].isin(gks_tracked)]

''' ***** Model Prep- This includes the creation of features within the model taken from Shots_Features.py  *****'''

# Importing model variables
model_vars = pdf.default_model_vars(shots=shots)

# Expected goals based on distance to goal and angle between the ball and the goal
b = pdf.params(model_vars)
model_vars["xg_basic"] = model_vars.apply(pdf.calculate_xG, b=b, axis=1)
# Storing goalkeeper distances for all tracked events
model_vars["gk_distance"] = shots.apply(pdf.dist_to_gk, track_df=track_df, axis=1)
# store distance in y axis from event to goalkeeper position in a dataframe
model_vars["gk_distance_y"] = shots.apply(pdf.y_to_gk, track_df=track_df, axis=1)
# store number of opposition's players closer than 3 meters in a dataframe
model_vars["close_players"] = shots.apply(pdf.three_meters_away, track_df=track_df, axis=1)
# store number of opposition's players inside a triangle in a dataframe
model_vars["triangle"] = shots.apply(pdf.players_in_triangle, track_df=track_df, axis=1)
# store opposition's goalkeeper distance to goal in a dataframe
model_vars["gk_dist_to_goal"] = shots.apply(pdf.gk_dist_to_goal, track_df=track_df, axis=1)
# create binary variable 1 if ball is closer to the goal than goalkeeper
model_vars["is_closer"] = np.where(model_vars["gk_dist_to_goal"] > model_vars["distance"], 1, 0)
# create binary variable 1 if header
model_vars["header"] = shots.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)

# store dependent variable in a numpy array
y = model_vars["goal"].values
# store independent variables in a numpy array
X = model_vars[
    ["x0", "is_closer", "angle", "distance", "gk_distance", "gk_distance_y", "triangle", "close_players", "header",
     "xg_basic"]].values

''' ***** Model Training- Calling in the model from Model.py  *****'''
# Calling in neural network from Model.py
X_train, X_val, X_cal, y_train, y_val, y_cal = Model.setup_model(X, y)
model, history = Model.run_model(X_train, y_train, X_val, y_val)

fig, axs = plt.subplots(2, figsize=(10, 12))

# plot training history - accuracy
axs[0].plot(history.history['accuracy'], label='train')
axs[0].plot(history.history['val_accuracy'], label='validation')
axs[0].set_title("Accuracy at each epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()

# plot training history - loss function
axs[1].plot(history.history['loss'], label='train')
axs[1].plot(history.history['val_loss'], label='validation')
axs[1].legend()
axs[1].set_title("Loss at each epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MSE")
plt.show()

# Model Assessment using ROC(+AUC- score 0.7-0.8 = acceptable, 0.8+ = good) + calibration curve
plt, axs = plt.subplots(2, figsize=(10, 12))
y_pred = model.predict(X_cal)
fpr, tpr, _ = roc_curve(y_cal, y_pred)
auc = roc_auc_score(y_cal, y_pred)
axs[0].plot(fpr, tpr, label="AUC = " + str(auc)[:4])
axs[0].plot([0, 1], [0, 1], color='black', ls='--')
axs[0].legend()
axs[0].set_ylabel('True Positive Rate')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_title('ROC curve')

# Calibration curve- Actual probability vs predicted probability
prob_true, prob_pred = calibration_curve(y_cal, y_pred, n_bins=10)
axs[1].plot(prob_true, prob_pred)
axs[1].plot([0, 1], [0, 1], color='black', ls='--')
axs[1].set_ylabel('Empirical Probability')
axs[1].set_xlabel('Predicted Probability')
axs[1].set_title("Calibration curve")
plt.show()
# Brier score- 0 represents perfect accuracy
print("Brier score", brier_score_loss(y_cal, y_pred))

# From our results we can see that our model is satisfactory, however it tends to assign more goals than in actuality
# when the probability of a goal is higher.
