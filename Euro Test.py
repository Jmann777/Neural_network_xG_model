import Statsbomb as Sb
import Shots_Features_Sb as pdf
from xG_advancement import create_model

# We are now going to test the model on Euro 2020 data
# Opening data
cid = 55
sid = 43
df_match_2 = Sb.sb_matches(cid, sid)
shots_2 = Sb.sb_shots_season(cid, sid)
track_df_2 = Sb.sb_tracking_season(cid, sid)
print(track_df_2.shape)

model = create_model()