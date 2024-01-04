import Model
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

model = Model.create_model()