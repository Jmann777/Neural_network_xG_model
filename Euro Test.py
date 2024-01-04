import Model
import numpy as np
import Statsbomb as Sb
import Shots_Features_Sb as pdf


# We are now going to test the model on Euro 2020 data
# Opening data
cid = 55
sid = 43
df_match = Sb.sb_matches(cid, sid)
shots = Sb.sb_shots_season(cid, sid)
track_df = Sb.sb_tracking_season(cid, sid)


# Filtering out non open_play shots
shots = shots[shots["sub_type_name"] == "Open Play"]
# Filter out shots where goalkeeper was not tracked
gks_tracked = track_df[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"]['id'].unique()
shots = shots[shots["id"].isin(gks_tracked)]

# Model variables
model_vars = pdf.default_model_vars(shots=shots)
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

model = Model.create_model()