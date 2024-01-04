"""
The script will look at training a shallow neural network with more advanced features to include opposition
player locations. It will work with ISL data
"""

# Importing modules
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import Statsbomb as Sb
import Shots_Features_Sb as pdf
import Model

#import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from keras.callbacks import EarlyStopping

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

''' ***** Model Prep- This includes the creation of features within the model and  *****'''

# Importing model variables
model_vars = pdf.default_model_vars(shots=shots)


# Logistic regression to calculate xg
def params(df):
    test_model = smf.glm(formula="goal_smf ~ angle + distance", data=df,
                         family=sm.families.Binomial()).fit()
    print(test_model.summary())
    return test_model.params


def calculate_xG(sh, b):
    bsum = b[0]
    for i, v in enumerate(["angle", "distance"]):
        bsum = bsum + b[i + 1] * sh[v]
    xG = 1 / (1 + np.exp(bsum))
    return xG


# Expected goals based on distance to goal and angle between the ball and the goal
b = params(model_vars)
model_vars["xg_basic"] = model_vars.apply(calculate_xG, b=b, axis=1)

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
# create binary varibale 1 if ball is closer to the goal than goalkeeper
model_vars["is_closer"] = np.where(model_vars["gk_dist_to_goal"] > model_vars["distance"], 1, 0)
# create binary variable 1 if header
model_vars["header"] = shots.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)

# store dependent variable in a numpy array
y = model_vars["goal"].values
# store independent variables in a numpy array
X = model_vars[
    ["x0", "is_closer", "angle", "distance", "gk_distance", "gk_distance_y", "triangle", "close_players", "header",
     "xg_basic"]].values

##############################################################################
# Training neural network - what is a neural network? https://www.youtube.com/watch?v=aircAruvnKk
# ----------------------------
# With the features created we can now train a neural network. We split the data 60% training, 20% validation and 20% test. Then, we scale inputs.
# As the next step, we create a neural network model. It follows similar design choices as Javier Fernandez's one. 2 dense layers sized 10 followed
# by a ReLU activation and a final layer size 1 with sigmoid activation to compute the probabilities. Our model optimizes the Brier score using Adam
# optimizer with learning rate 0.001 default betas. We use as suggested early stopping with minimum delta 1e-5 and batch size 16. However, we also use patience
# equal to 50 not to stop the first time when the validation loss is not changing.

# Split the data into a train, validation, and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=123, stratify=y)
X_cal, X_val, y_cal, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=123, stratify=y_test)

# Scale the data to ensure equality of feature contribution
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_cal = scaler.transform(X_cal)


# Model creation- see model.py
model = Model.create_model()
# early stopping object (callback)- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
callback = EarlyStopping(min_delta=1e-5, patience=50, mode='min', monitor='val_loss', restore_best_weights=True)
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=1, batch_size=16, callbacks=[callback])

fig, axs = plt.subplots(2, figsize=(10,12))


#plot training history - accuracy
axs[0].plot(history.history['accuracy'], label='train')
axs[0].plot(history.history['val_accuracy'], label='validation')
axs[0].set_title("Accuracy at each epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()

#plot training history - loss function
axs[1].plot(history.history['loss'], label='train')
axs[1].plot(history.history['val_loss'], label='validation')
axs[1].legend()
axs[1].set_title("Loss at each epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MSE")
plt.show()

# Model Assessment using ROC(+AUC- score 0.7-0.8 = acceptable, 0.8+ = good) + calibration curve
fig, axs = plt.subplots(2, figsize=(10,12))
y_pred = model.predict(X_cal)
fpr, tpr, _ = roc_curve(y_cal, y_pred)
auc = roc_auc_score(y_cal, y_pred)
axs[0].plot(fpr,tpr,label= "AUC = " + str(auc)[:4])
axs[0].plot([0, 1], [0, 1], color='black', ls = '--')
axs[0].legend()
axs[0].set_ylabel('True Positive Rate')
axs[0].set_xlabel('False Positive Rate')
axs[0].set_title('ROC curve')

# Calibration curve- Actual probability vs predicted probability
prob_true, prob_pred = calibration_curve(y_cal, y_pred, n_bins=10)
axs[1].plot(prob_true, prob_pred)
axs[1].plot([0, 1], [0, 1], color='black', ls = '--')
axs[1].set_ylabel('Empirical Probability')
axs[1].set_xlabel('Predicted Probability')
axs[1].set_title("Calibration curve")
plt.show()
# Brier score- 0 represents perfect accuracy
print("Brier score", brier_score_loss(y_cal, y_pred))

# From our results we can see that our model is satisfactory, however it tends to assign more goals than in actuality
# when the probability of a goal is higher. See Euro Test.py for use of model
