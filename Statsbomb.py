from mplsoccer import Sbopen
import numpy as np
import pandas as pd


def sb_matches(cid, sid):
    parser = Sbopen()
    df_match = parser.match(competition_id=cid, season_id=sid)
    matches = df_match.match_id.unique()
    np.transpose(matches)
    return matches


def sb_shots_season(cid, sid):
    parser = Sbopen()
    df_match = parser.match(competition_id=cid, season_id=sid)
    matches = df_match.match_id.unique()
    shot_df = pd.DataFrame()
    for match in matches:
        parser = Sbopen()
        df_event = parser.event(match)[0]
        shots = df_event.loc[df_event["type_name"] == "Shot"]
        shots.x = shots.x.apply(lambda cell: cell * 105 / 120)
        shots.y = shots.y.apply(lambda cell: cell * 68 / 80)
        shot_df = pd.concat([shot_df, shots], ignore_index=True)
    shot_df.reset_index(drop=True, inplace=True)
    return shot_df


def sb_tracking_season(cid, sid):
    parser = Sbopen()
    df_match = parser.match(competition_id=cid, season_id=sid)
    matches = df_match.match_id.unique()
    track_df = pd.DataFrame()
    for match in matches:
        parser = Sbopen()
        df_track = parser.event(match)[2]
        df_track.x = df_track.x.apply(lambda cell: cell * 105 / 120)
        df_track.y = df_track.y.apply(lambda cell: cell * 68 / 80)
        track_df = pd.concat([track_df, df_track], ignore_index=True)
    track_df.reset_index(drop=True, inplace=True)
    return track_df


def sb_events_season(cid, sid):
    parser = Sbopen()
    df_match = parser.match(competition_id=cid, season_id=sid)
    matches = df_match.match_id.unique()
    event_df = pd.DataFrame()
    for match in matches:
        parser = Sbopen()
        df_event = parser.event(match)[0]
        df_event.x = df_event.x.apply(lambda cell: cell * 105 / 120)
        df_event.y = df_event.y.apply(lambda cell: cell * 68 / 80)
        event_df = pd.concat([event_df, df_event], ignor_index=True)
    event_df.reset_index(drop=True, inplace=True)
    return event_df
