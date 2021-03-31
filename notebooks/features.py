import pandas as pd
import numpy as np
import os

project_dir = os.getcwd().split('\\')[:-1]
project_dir = '\\'.join(project_dir)
data_dir = project_dir + '\\data'
video_dir = data_dir+'\\interim\\video'
video_files = [file for file in os.listdir(video_dir)]

def compute_head_motion(video_data):
    processed = {}
    cols = [col for col in video_data.columns if col.startswith('pose')]
    for col in cols:
        processed[f'std_{col}'] = video_data[col].std()
    cols = [col for col in video_data.columns if col.startswith('pose_T')]
    processed['compound_Motion'] = video_data[cols].std().mean()
    return processed

def compute_emotions(video_data):
    """Compute emotions based on Imotions."""
    processed = {}
    processed['mean_Happy'] = ((video_data['AU06_r'] + video_data['AU12_r'])/2).mean()
    processed['mean_Sad'] = ((video_data['AU01_r'] + video_data['AU04_r'] + video_data['AU15_r'])/3).mean()
    processed['mean_Angry'] = ((video_data['AU04_r'] + video_data['AU05_r'] + video_data['AU07_r'] + video_data['AU23_r'])/4).mean()
    processed['mean_Scared'] = ((video_data['AU01_r'] + video_data['AU02_r'] + video_data['AU04_r'] +video_data['AU05_r'] +
                                 video_data['AU07_r'] + video_data['AU20_r'] + video_data['AU26_r'])/7).mean()
    return processed

def compute_arousal(video_data):
    processed = {}
    AU_cols = [col for col in video_data.columns if col.startswith('AU') and col.endswith('_r') and '45' not in col]
    df = video_data[AU_cols] - video_data[AU_cols].rolling(60*25, min_periods=1).mean()
    df[df<0] = 0
    processed['mean_Arousal'] = (df.apply(pd.Series.nlargest, axis=1, n=5)).mean(axis=1).mean()
    return processed

def compute_mean_AUs(video_data):
    processed = {}
    AU_cols = [col for col in video_data.columns if col.startswith('AU') & col.endswith('_r')]
    for AU in AU_cols:
        processed[f'mean_{AU[:-2]}'] = video_data[AU].mean()
        processed[f'mean_change_{AU[:-2]}'] = (video_data[AU] - video_data[AU].rolling(60*25, min_periods=1).mean()).mean()
    return processed

def compute_std_AUs(video_data):
    processed = {}
    AU_cols = [col for col in video_data.columns if col.startswith('AU') & col.endswith('_r')]
    for AU in AU_cols:
        processed[f'std_{AU[:-2]}'] = video_data[AU].std()
    return processed


def compute_blink_rate(video_data):
    blinks = sum(video_data['AU45_c'].diff() == 1)
    seconds = video_data['t_from_start'].values[-1] - video_data['t_from_start'].values[0]
    blinks_sec = blinks/seconds
    blinks_min = blinks_sec*60
    return blinks_min


def compute_percentage_EC(video_data):
    """Compute the amount of percentage of frames where the eyes are closed."""
    per_EC = video_data['AU45_c'].sum()/len(video_data['AU45_c'])
    return per_EC


def compute_distance(X_1, X_2, Y_1, Y_2):
    return np.sqrt(((X_1-X_2)**2)+((Y_1-Y_2)**2))


def compute_distance_for_points(points, video_data, video_data_col):
    for point in points:
        video_data_col += compute_distance(video_data[f'eye_lmk_X_{point[0]}'], video_data[f'eye_lmk_X_{point[1]}'],
                                           video_data[f'eye_lmk_Y_{point[0]}'], video_data[f'eye_lmk_Y_{point[1]}'])
    return video_data_col/len(points)


def compute_avg_PD_2_eyes(video_data):

    left_eyes = [(27,23), (26, 22), (25, 21), (24, 20)]
    right_eyes = [(55,51), (54, 50), (53, 49), (52, 48)]
    
    
    video_data.loc[:,'avg_PD_left'] = 0
    video_data.loc[:,'avg_PD_left'] = compute_distance_for_points(left_eyes, video_data, video_data.loc[:,'avg_PD_left'])

    video_data.loc[:,'avg_PD_right'] = 0
    video_data.loc[:,'avg_PD_right'] = compute_distance_for_points(right_eyes, video_data, video_data.loc[:,'avg_PD_right'])
    
                                                                      
    video_data.loc[:,'PD'] = (video_data.loc[:,'avg_PD_left'] + video_data.loc[:,'avg_PD_right']) / 2
    return video_data


def compute_PD_features(video_data):
    processed = {}
    video_data = compute_avg_PD_2_eyes(video_data)
    
    processed['mean_PD'] = np.mean(video_data['PD'])
    processed['std_PD'] = np.std(video_data['PD'])
    processed['max_PD'] = np.max(video_data['PD'])
    
    return processed