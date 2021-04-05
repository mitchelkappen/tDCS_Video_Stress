## Import the necessary packages
import pandas as pd
import numpy as np
import os


def compute_head_motion(video_data: pd.DataFrame) -> dict:
    """Computes various measures for head motion and retuns these in a dictionary."""
    processed = {} # Create the dict to store the results
    cols = [col for col in video_data.columns if col.startswith('pose')] # Get the columns relevant for head pose
    for col in cols: # For each of the relevant columns
        processed[f'std_{col}'] = video_data[col].std() # Computes standard deviation for each of the 6 pose cols
    cols = [col for col in video_data.columns if col.startswith('pose_T')] # Get the 3 translation pose collumns
    processed['compound_Motion'] = video_data[cols].std().mean() # Compute the mean of the std of all 3 cols 
    return processed # Return the results in a dict


def compute_emotions(video_data: pd.DataFrame) -> dict:
    """Computes emotions based on Imotions and returns these in a dict."""
    processed = {} # Create the dict to store the results
    ## Compute the average emotion, using the relevant FAUs and store these in the dict
    processed['mean_Happy'] = ((video_data['AU06_r'] + video_data['AU12_r'])/2).mean()
    processed['mean_Sad'] = ((video_data['AU01_r'] + video_data['AU04_r'] + video_data['AU15_r'])/3).mean()
    processed['mean_Angry'] = ((video_data['AU04_r'] + video_data['AU05_r'] + video_data['AU07_r'] + video_data['AU23_r'])/4).mean()
    processed['mean_Scared'] = ((video_data['AU01_r'] + video_data['AU02_r'] + video_data['AU04_r'] +video_data['AU05_r'] +
                                 video_data['AU07_r'] + video_data['AU20_r'] + video_data['AU26_r'])/7).mean()
    return processed # Return the results in a dict


def compute_arousal(video_data: pd.DataFrame) -> dict:
    """Computes the arousal based on the mean intensity (_r) of all FAUs (except 45) and returns this in a dict. Based on the Noldus whitepaper."""
    processed = {} # Create the dict to store the results
    
    AU_cols = [col for col in video_data.columns if col.startswith('AU') and col.endswith('_r') and '45' not in col] # Get all the columns that reflect intensity (_r) FAUs except FAU45
    df = video_data[AU_cols] - video_data[AU_cols].rolling(60*25, min_periods=1).mean() # Normalise the FAU intensity based on the previous 60 seconds mean intensity
    df[df<0] = 0 # Set all negative intensities to zero
    processed['mean_Arousal'] = (df.apply(pd.Series.nlargest, axis=1, n=5)).mean(axis=1).mean() # Compute the arousal as the mean on the 5 FAUs with the most intensity
    return processed # Return the results in a dict


def compute_mean_AUs(video_data: pd.DataFrame) -> dict:
    """Computes the mean intensity and mean change in intensity for all FAUs and returns this in a dict."""
    processed = {} # Create the dict to store the results
    AU_cols = [col for col in video_data.columns if col.startswith('AU') & col.endswith('_r')] # Get FAU intensity columns 
    for AU in AU_cols: # For each FAU intensity col
        processed[f'mean_{AU[:-2]}'] = video_data[AU].mean() # Compute the mean intensity
        processed[f'mean_change_{AU[:-2]}'] = (video_data[AU] - video_data[AU].rolling(60*25, min_periods=1).mean()).mean() # Compute the mean change, reflected as the mean of the corrected intensity
    return processed # Return the results in a dict


def compute_std_AUs(video_data: pd.DataFrame) -> dict:
    """Computes the standard deviation of intensity for all FAUs and returns this in a dict."""
    processed = {} # Create the dict to store the results
    AU_cols = [col for col in video_data.columns if col.startswith('AU') & col.endswith('_r')] # Get FAU intensity columns 
    for AU in AU_cols: # For each FAU intensity col
        processed[f'std_{AU[:-2]}'] = video_data[AU].std() # Compute the standard deviation of the intensity
    return processed # Return the results in a dict


def compute_blink_rate(video_data: pd.DataFrame) -> float:
    """Compute the amount of blinks per minute (blink_rate) for a given video signal and returns the result as a float."""
    blinks = sum(video_data['AU45_c'].diff() == 1) # Get the number of blinks using the AU45_c column
    seconds = video_data['t_from_start'].values[-1] - video_data['t_from_start'].values[0] # Get the amount of seconds in the given video signal
    blinks_sec = blinks/seconds # Compute the amount of blinks per seconds
    blinks_min = blinks_sec*60 # Compute the amount of blinks per minute
    return blinks_min # Return the blink rate


def compute_percentage_EC(video_data: pd.DataFrame) -> float:
    """Computes the amount of percentage of frames where the eyes are closed and returns the result as a float"""
    per_EC = video_data['AU45_c'].sum()/len(video_data['AU45_c']) # Compute how many of the frames, the eyes were classified as closed
    return per_EC # Return this result


def compute_distance(X_1, X_2, Y_1, Y_2) -> float:
    """Computes the distance between two points in a 2-dimensional space, and returns this distance as a float"""
    return np.sqrt(((X_1-X_2)**2)+((Y_1-Y_2)**2))


def compute_distance_for_points(points: list, video_data: pd.DataFrame, video_data_col: pd.DataFrame) -> pd.DataFrame:
    """Computes the mean PD as the average of distance between all pairs of opposite points on one pupil, and does this for each frame and returns this as a data column."""
    for point in points: # For each pair of opposite points
        video_data_col += compute_distance(video_data[f'eye_lmk_X_{point[0]}'], video_data[f'eye_lmk_X_{point[1]}'],
                                           video_data[f'eye_lmk_Y_{point[0]}'], video_data[f'eye_lmk_Y_{point[1]}']) # Compute the distance and add this to the previous computed distance
    return video_data_col/len(points) # Compute the average distance by dividing through number of pairs of opposite points


def compute_avg_PD_2_eyes(video_data: pd.DataFrame) -> pd.DataFrame:
    """For both eyes, compute the pupil diameter, as the average of 4 lines across the pupil. Return a column, which holds the pupil diameter (as the average of the PD between both eyes) for each frame in the video signal."""
    
    ## First, we need the points on the pupils that are opposite of each other
    ## These are stored as a list of tuples, where each tuple is pair of opposite points
    left_eyes = [(27,23), (26, 22), (25, 21), (24, 20)] # For the left eye
    right_eyes = [(55,51), (54, 50), (53, 49), (52, 48)] # For the right eye
    
    
    video_data.loc[:,'avg_PD_left'] = 0 # We create a column to store the average PD in each frame for the left eye 
    video_data.loc[:,'avg_PD_left'] = compute_distance_for_points(left_eyes, video_data, video_data.loc[:,'avg_PD_left']) # Compute the average PD in each frame for the left eye

    video_data.loc[:,'avg_PD_right'] = 0 # We create a column to store the average PD in each frame for the left eye 
    video_data.loc[:,'avg_PD_right'] = compute_distance_for_points(right_eyes, video_data, video_data.loc[:,'avg_PD_right']) # Compute the average PD in each frame for the left eye
    
                                                                      
    video_data.loc[:,'PD'] = (video_data.loc[:,'avg_PD_left'] + video_data.loc[:,'avg_PD_right']) / 2 # Create a new column that holds the average of the PD for both eyes
    return video_data # Return the entire DataFrame


def compute_PD_features(video_data: pd.DataFrame) -> dict:
    """Computes various features concerning the pupil diameter (PD). Returns these results in a dict."""
    processed = {} # Create the dict to store the results
    video_data = compute_avg_PD_2_eyes(video_data) # Compute the average pupil diameter for both eyes for each frame
    
    processed['mean_PD'] = np.mean(video_data['PD']) # Compute the average PD throughout the signal
    processed['std_PD'] = np.std(video_data['PD']) # Compute the standard deviation of the PD throughout the signal
    processed['max_PD'] = np.max(video_data['PD']) # Compute the max PD throughout the signal
    
    return processed # Return the results in a dict