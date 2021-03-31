## Import the required modules
import os
import subprocess
# Store the required directories in variables so we can easily access them
project_dir = os.getcwd().split('\\')[:-1]
project_dir = '\\'.join(project_dir) # Makes sure the script can be used across different computers by setting the working directory dynamically
video_dir = project_dir + "\\tDCS_Video_Stress\\data\\raw\\Video"
video_files = os.listdir(video_dir) # Get all the videos that are currently available for extraction
processed_dir = project_dir + '\\tDCS_Video_Stress\\data\\raw\\Video_features'
processed_files = os.listdir(processed_dir) # Get all the files that have allready been processed by OpenFace
processed_ppts = [file[4:7] for file in processed_files if file.endswith('.csv')] # Extract the participants ids from the files

for video in video_files: # Run through all video files
    ppt = video[4:7] # Get the participant id from the title of the video
    if ppt in processed_ppts: # If the current participant id is in the list of processed participants than skip the current participant 
        continue
    else: # Process the current video using the feature extraction module from OpenFace
        video_loc = video_dir + '\\' + video
        subprocess.run(['D:\\Documenten\\Artificial Intelligence Master\\Semester 3\\Internship Mitch\\OpenFace\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe',
                        '-f', video_loc, "-2Dfp",   "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze", "-hogalign"])

