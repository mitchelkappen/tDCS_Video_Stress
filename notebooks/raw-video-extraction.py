
import os
import subprocess
project_dir = os.getcwd().split('\\')[:-1]
project_dir = '\\'.join(project_dir)

video_dir = project_dir + "\\tDCS_Video_Stress\\data\\raw\\Video"
video_files = os.listdir(video_dir)
processed_dir = project_dir + '\\tDCS_Video_Stress\\processed'
processed_files = os.listdir(processed_dir)
processed_ppts = [file[4:7] for file in processed_files if file.endswith('.csv')]

openface_loc = 'D:\\Documenten\\Artificial Intelligence Master\\Semester 3\\Internship Mitch\\OpenFace\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe'

for video in video_files:
    ppt = video[4:7]
    if ppt in processed_ppts:
        continue
    else:
        video_loc = video_dir + '\\' + video
        subprocess.run(['D:\\Documenten\\Artificial Intelligence Master\\Semester 3\\Internship Mitch\\OpenFace\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe',
                        '-f', video_loc, "-2Dfp",   "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze", "-hogalign"])

