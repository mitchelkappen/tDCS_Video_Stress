import pandas as pd
import numpy as np
import os
import csv
import neurokit2 as nk

project_dir = os.getcwd().split('\\')[:-1]
project_dir = '\\'.join(project_dir)
data_dir = project_dir + '\\data'
physio_dir = data_dir+'\\interim\\physiological'
physio_files = [file for file in os.listdir(physio_dir)]
baseline_dir = data_dir + '\\interim\\physiological_baseline'
baseline_files = [file for file in os.listdir(baseline_dir)]

def compute_PP_EDA(physio_data, pp):
    processed = {}
    
    freq=round(1/(physio_data.t_from_start.values[1] - physio_data.t_from_start.values[0]))
    seconds = physio_data.t_from_start.values[-1] - physio_data.t_from_start.values[0]
    
    signals, info = nk.eda_process(physio_data.raw_EDA, sampling_rate=freq)
    
    processed['mean_SCL'] = signals.EDA_Tonic.mean()
    processed['std_SCL'] = signals.EDA_Tonic.std()
    processed['max_SCL'] = signals.EDA_Tonic.max()
    processed['min_SCL'] = signals.EDA_Tonic.min()
    processed['mean_SCL_Baseline'] = np.concatenate((signals.EDA_Tonic.values[:30*freq], signals.EDA_Tonic.values[-30*freq:])).mean() # mean of first 30 and last 30 seconds of EDA as baseline
    processed['pp'] = pp

    return processed

def compute_EDA_Targets(physio_data: pd.DataFrame) -> dict:
    """Computes the EDA Target variables from an abritray dataframe containing the raw EDA signal. Returns these target variables in a dict."""
    
    pp = physio_data.pp.values[0] # Get the pp id
    
    ## Open the baseline, and all descriptive files, then get the value that corresponds to the current pp id and store these in corresponding variables
    ## Baseline descriptives
    mean_Baseline = pd.read_csv(data_dir +  '\\information\\SCL stats\\PP_meanSCL_Baseline.csv', index_col=0)['mean_SCL'].to_dict()[pp]
    min_Baseline = pd.read_csv(data_dir +  '\\information\\SCL stats\\PP_minSCL_Baseline.csv', index_col=0)['min_SCL'].to_dict()[pp]
    ## All Descriptives
    mean_all = pd.read_csv(data_dir +  '\\information\\SCL stats\\PP_meanSCL_all.csv', index_col=0)['mean_SCL'].to_dict()[pp]
    std_all = pd.read_csv(data_dir +  '\\information\\SCL stats\\PP_stdSCL_all.csv', index_col=0)['std_SCL'].to_dict()[pp]
    max_all = pd.read_csv(data_dir +  '\\information\\SCL stats\\PP_maxSCL_all.csv', index_col=0)['max_SCL'].to_dict()[pp]
    
    processed = {} # Create the dict to store the target variables in and which is returned later in the function
    
    df = physio_data.dropna() # Drop the empty rows in the function (this occurs due to differences in sampling frequency between ECG and EDA signal in the AMSDATA files)
    
    freq=round(1/(df.t_from_start.values[1] - df.t_from_start.values[0])) # Calculate the frequency at which the raw EDA signal was sampled 
    seconds = df.t_from_start.values[-1] - df.t_from_start.values[0] # Calculate the amount seconds the physio signal contains
    
    signals, info = nk.eda_process(physio_data.raw_EDA.dropna(), sampling_rate=freq) # Process the EDA signals using the eda_process function from NeuroKit2
    
    ## Compute various EDA target variables and store these in the dict. Also add the pp id to the dict and return this dict
    processed['mean_SCL'] = signals.EDA_Tonic.mean() # Compute the mean SCL signal
    processed['corrected_mean_SCL'] = processed['mean_SCL'] - mean_Baseline # Compute the corrected mean SCL, by subtracting the mean SCL from the baseline component
    processed['range_corrected_mean_SCL'] = ((signals.EDA_Tonic - min_Baseline) / (max_all - min_Baseline)).mean() # Compute the range corected min SCL, by Min Max scaling the signal and before computing the mean
    processed['standardised_mean_scl'] = ((signals.EDA_Tonic - mean_all) / std_all).mean() # Compute the standarised mean SCL, by standardising using the mean and std SCL throughout all the active components
    processed['frequency_NS_SCR'] = len(info['SCR_Peaks'])/seconds * 60 # Compute the frequency in which Non-Stimulus Skin Conductance Responses (NS-SCRs) occured, which is a EDA measures related to the Phasic component
    return processed

def compute_ECG_Targets(physio_data: pd.DataFrame) -> dict:
    """Computes the EDA Target variables from an abritray dataframe containing the raw EDA signal. Returns these target variables in a dict."""
    
    pp = physio_data.pp.values[0] # Get the pp id
    
    # Open the baseline HRV measures DataFrames, and then get the value that corresponds to the current pp id and store these in corresponding variables
    HRV_RMSSD_baseline = pd.read_csv(data_dir +  '\\information\\HRV stats\\PP_RMSSD_Baseline.csv', index_col=0)['HRV_RMSSD'].to_dict()[pp]
    HRV_MeanNN_baseline = pd.read_csv(data_dir +  '\\information\\HRV stats\\PP_MeanNN_Baseline.csv', index_col=0)['HRV_MeanNN'].to_dict()[pp]
    HRV_SDNN_baseline = pd.read_csv(data_dir +  '\\information\\HRV stats\\PP_SDNN_Baseline.csv', index_col=0)['HRV_SDNN'].to_dict()[pp]    
    
    processed = {} # Create the dict to store the target variables in and which is returned later in the function
    
    freq=round(1/(physio_data.t_from_start.values[1] - physio_data.t_from_start.values[0])) # Calculate the frequency at which the raw EDA signal was sampled 
    seconds = physio_data.t_from_start.values[-1] - physio_data.t_from_start.values[0] # Calculate the amount seconds the physio signal contains
    
    ecg_signals, info = nk.ecg_process(physio_data["raw_ECG"], sampling_rate=freq) # Get the clean ECG signal using the ecg_process function from neurokit2    
    ecg_features = nk.ecg_intervalrelated(ecg_signals) # Compute HRV measures using the ecg_intervalrelated function from NeuroKit2
    
    ## Compute various HRV target variables and store these in the dict. Also add the pp id to the dict and return this dict
    processed['HRV_MeanNN'] = ecg_features['HRV_MeanNN'].values[0] # Store the mean of the time interval between the NN peaks
    processed['HRV_RMSSD'] = ecg_features['HRV_RMSSD'].values[0] # Store the Root Mean Squared Standard deviation of the time interval between the NN peaks 
    processed['HRV_SDNN'] = ecg_features['HRV_SDNN'].values[0] # Store the standard deviation of the time interval between the NN peaks
    processed['HRV_MeanNN_corrected'] = processed['HRV_MeanNN'] - HRV_MeanNN_baseline # Correct the meanNN by subtracting with the meanNN from the baseline component and store this 
    processed['HRV_RMSSD_corrected'] = processed['HRV_RMSSD'] - HRV_RMSSD_baseline # Correct the RMSSD by subtracting with the RMSSD from the baseline component and store this
    processed['HRV_SDNN_corrected'] = processed['HRV_SDNN'] - HRV_SDNN_baseline # Correct the SDNN by subtracting with the SDNN from the baseline component and store this
    return processed