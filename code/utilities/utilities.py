"""
Script Title: Utilities
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script contains utility functions that are used in the radar tracking pipeline.
"""

import os
import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
import plotting
from utilities.check_start_and_stop import RectangleA, RectangleB, RectangleC, RectangleD, RectangleE, RectangleF

def find_files(root,txt_filename):
    """
    Finds the files in the root directory that are given in the txt file
    """
    with open(txt_filename, 'r') as f:
        lines = f.readlines()
    files = []
    for line in lines:
        files.append(line[:-1])

    path_list = []
    for item in os.listdir(root):
        list_of_files = glob.glob(os.path.join(root, item, '*.json'))
        for file in list_of_files:
            if  os.path.basename(file) in files:
                path_list.append(file)

    return path_list

def write_filenames_to_txt(filename, txt_filename):
    """
    Writes the filename to the txt file, if the filename is not already written
    """
    with open(txt_filename, 'r') as f:
        lines = f.readlines()
    files = []
    for line in lines:
        files.append(line[:-1])

    already_written = False
    if os.path.basename(filename) in files:
        already_written = True
        print(f"File {os.path.basename(filename)} already written to txt file")

    if not already_written:
        with open(txt_filename, 'a') as f:
            f.write(os.path.basename(filename) + "\n")
   
def make_new_directory(wokring_directory):
    """
    Makes a new directory for the radar tracking results, and a subdirectory for the current date, so that the results are saved in a structured way.
    """
    root = ""
    for i in range(len(wokring_directory.split("/"))-1):
        root += wokring_directory.split("/")[i] + "/"
    root += "radar_tracking_results"

    if not os.path.exists(root):
        os.mkdir(root)
        print(f"Directory {root} created")
    else:
        print(f"Directory {root} already exists")

    todays_date = datetime.datetime.now().strftime("%d-%b")
    path = os.path.join(root,todays_date)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def check_timegaps(timestamps):
    """
    Checks if the time gaps between the timestamps are consistent
    """
    time_gaps = []
    for k in range(len(timestamps)-1):
        time_gap = timestamps[k+1]-timestamps[k]
        time_gaps.append(time_gap)

    time_gap = np.mean(time_gaps)
    for k in range(len(time_gaps)):
        if  not (0.8*time_gap < time_gaps[k][0] < 1.2*time_gap):
            print(f"Time gap are not consistent, time gap {time_gaps[k][0]} at index {k}")

def check_speed_of_tracks(unvalid_tracks, track_history):
    """
    If the average speed of a track is higher than 6 knots, the track is considered invalid.+
    """
    track_lengths_dict = {}
    for track in track_history.items():
        first_mean, cov = track[1][0].posterior
        last_mean, cov = track[1][-1].posterior
        first_point = (first_mean[0],first_mean[2])
        last_point = (last_mean[0],last_mean[2])
        track_length = np.sqrt((first_point[0]-last_point[0])**2 + (first_point[1]-last_point[1])**2)
        #print(f"lenght of track {track[0]} = {track_length:.2f}") 
        track_lengths_dict[track[0]] = track_length

    for track in track_history.items():
        track_id = track[0]
        track_start_time = track[1][0].timestamp
        track_end_time = track[1][-1].timestamp
        track_time = track_end_time - track_start_time
        track_speed = track_lengths_dict[track_id]/track_time
        track_speed_knots = 1.94384449*track_speed
        # print(f"Speed of track {track_id} = {track_speed_knots:.2f} knots")
        if track_speed_knots > 6 and track_id not in unvalid_tracks:
            unvalid_tracks.append(track_id)

    return unvalid_tracks, track_lengths_dict

def check_lenght_of_tracks(unvalid_tracks, track_lengths_dict):
    """
    Adds tracks with a length less than 10 meters to the list of unvalid tracks
    """
    for track_length in track_lengths_dict.items():
        if track_length[1] < 10 and track_length[0] not in unvalid_tracks:
            unvalid_tracks.append(track_length[0])
    return unvalid_tracks

def check_coherence_factor(track_history,coherence_factor=0.75):
    """
    Checks the coherence factor of the tracks, and adds the tracks with a coherence factor less than the given value to the list of unvalid tracks
    """
    not_valid_tracks = []
    for track in track_history.items():
        u = []
        v = []
        last_mean, cov = track[1][0].posterior
        for k,track_point in enumerate(track[1]):
            if k == 0:
                continue
            mean, cov = track_point.posterior
            u.append([[mean[0]-last_mean[0]],[mean[2]-last_mean[2]]])
            v.append([mean[1],mean[3]])
            last_mean = mean
        u = np.array(u)
        v = np.array(v)

        ck = 0
        for k in range(1,len(u)):
            c_k = np.dot(np.transpose(v[k]),u[k])/(np.linalg.norm(u[k])*np.linalg.norm(v[k]))
            ck += c_k[0]

        if ck/len(u) < coherence_factor:
            not_valid_tracks.append(track[0])

    return not_valid_tracks

def print_current_tracks(track_history):
    """
    Prints the current tracks
    """
    print(f"Number of tracks = {len(track_history)}")
    for key, value in track_history.items():
        print(f"Track {key}")
    print("\n")

def check_if_track_is_stationary(track):
    """
    Checks if the track is stationary
    """
    first_mean, cov = track[1][0].posterior
    last_mean, cov = track[1][-1].posterior
    distance_travelled = 0
    for i in range(1,len(track[1])-1):
        mean, cov = track[1][i].posterior
        last_mean, cov = track[1][i-1].posterior
        x = mean[0]-last_mean[0]
        y = mean[2]-last_mean[2]
        distance_travelled += np.sqrt(x**2 + y**2)
    
    first_mean, cov = track[1][0].posterior
    last_mean, cov = track[1][-1].posterior
    x = first_mean[0]-last_mean[0]
    y = first_mean[2]-last_mean[2]
    distance_from_start_to_end = np.sqrt(x**2 + y**2)

    track_id = track[0]
    track_start_time = track[1][0].timestamp
    track_end_time = track[1][-1].timestamp
    track_time = track_end_time - track_start_time
    track_speed = distance_from_start_to_end/track_time
    track_speed_knots = 1.94384449*track_speed
    # print(f"Duration of track {track_id} = {track_time:.2f} seconds")
    # print(f"Distance from start to end of track {track_id} = {distance_from_start_to_end:.2f} meters")
    # print(f"Speed of track {track_id} = {track_speed_knots:.2f} knots")
    # print("\n")
    if (distance_from_start_to_end < 30) or (distance_travelled < 30):
        return True
    else:
        return False
    


def histogram_of_tracks_duration(wokring_directory, track_history, reset=False):
    """
    Saves the duration of the tracks in a npy file, which later can be used to plot a histogram of the track duration
    """
    npy_file = f"{wokring_directory}/code/npy_files/track_duration.npy"
    if not os.path.exists(npy_file) or reset:
        tracks_duration_dict = {}
        tracks_duration_dict["0-20"] = [0,0]
        tracks_duration_dict["20-40"] = [0,0]
        tracks_duration_dict["40-60"] = [0,0]
        tracks_duration_dict["60-80"] = [0,0]
        tracks_duration_dict["80-100"] = [0,0]
        tracks_duration_dict["100-120"] = [0,0]
        tracks_duration_dict["120-140"] = [0,0]
        tracks_duration_dict["140-160"] = [0,0]
        tracks_duration_dict["160-180"] = [0,0]
        tracks_duration_dict["180-200"] = [0,0]
        tracks_duration_dict[">200"] = [0,0]
        np.save(npy_file,tracks_duration_dict)

    tracks_duration_dict = np.load(npy_file,allow_pickle=True).item()

    track_durations = []
    for track in track_history.items():
        track_start_time = track[1][0].timestamp
        track_end_time = track[1][-1].timestamp
        track_time = track_end_time - track_start_time
        if check_if_track_is_stationary(track):
            track_durations.append([track_time,1])
        else:
            track_durations.append([track_time,0])

    for duration,stationary in track_durations:
        if duration < 20:
            if stationary:
                tracks_duration_dict["0-20"][1] += 1
            else:
                tracks_duration_dict["0-20"][0] += 1

        elif 20 <= duration < 40:
            if stationary:
                tracks_duration_dict["20-40"][1] += 1
            else:
                tracks_duration_dict["20-40"][0] += 1
        elif 40 <= duration < 60:
            if stationary:
                tracks_duration_dict["40-60"][1] += 1
            else:
                tracks_duration_dict["40-60"][0] += 1
        elif 60 <= duration < 80:
            if stationary:
                tracks_duration_dict["60-80"][1] += 1
            else:
                tracks_duration_dict["60-80"][0] += 1
        elif 80 <= duration < 100:
            if stationary:
                tracks_duration_dict["80-100"][1] += 1
            else:
                tracks_duration_dict["80-100"][0] += 1
        elif 100 <= duration < 120:
            if stationary:
                tracks_duration_dict["100-120"][1] += 1
            else:
                tracks_duration_dict["100-120"][0] += 1
        elif 120 <= duration < 140:
            if stationary:
                tracks_duration_dict["120-140"][1] += 1
            else:
                tracks_duration_dict["120-140"][0] += 1
        elif 140 <= duration < 160:
            if stationary:
                tracks_duration_dict["140-160"][1] += 1
            else:
                tracks_duration_dict["140-160"][0] += 1
        elif 160 <= duration < 180:
            if stationary:
                tracks_duration_dict["160-180"][1] += 1
            else:
                tracks_duration_dict["160-180"][0] += 1
        elif 180 <= duration < 200:
            if stationary:
                tracks_duration_dict["180-200"][1] += 1
            else:
                tracks_duration_dict["180-200"][0] += 1
        else:
            if stationary:
                tracks_duration_dict[">200"][1] += 1
            else:
                tracks_duration_dict[">200"][0] += 1

    np.save(f"{wokring_directory}/code/npy_files/track_duration.npy",tracks_duration_dict)
    #rint(tracks_duration_dict)

def plot_histogram_of_tracks_duration(wokring_directory):
    """
    Plots a histogram of the track duration
    """
    tracks_duration_dict = np.load(f"{wokring_directory}/code/npy_files/track_duration.npy",allow_pickle=True).item()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Get a list of colors for each bar
    #colors = plt.cm.viridis(np.linspace(0, 1, len(tracks_duration_dict)))

    data1 = [value[0] for value in tracks_duration_dict.values()]
    data2 = [value[1] for value in tracks_duration_dict.values()]
    ax.bar(tracks_duration_dict.keys(), data1, color='#1f77b4', label='Moving tracks')
    ax.bar(tracks_duration_dict.keys(), data2, color='#2ca02c',bottom=data1, label='Stationary tracks')
    ax.set_xlabel('Duration of tracks [s]',fontsize=15)
    ax.set_ylabel('Number of tracks',fontsize=15)
    ax.legend(fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f"{os.path.dirname(wokring_directory)}/radar_tracking_results/histogram_track_duration.png",dpi=400)
    print(f"Saved histogram of track duration to {os.path.dirname(wokring_directory)}/radar_tracking_results/histogram_track_duration.png")
    plt.close()

def read_out_txt_file(root):
    """
    Reads out the txt file with the timestamps of the files that should be skipped, related to remove_files
    """
    txt_file = glob.glob(os.path.join(root, '*.txt'))
    timestamps = []
    with open(txt_file[0], 'r') as f:
        lines = f.readlines()
        for line in lines:
            timestamps.append(line.split()[0])
    return timestamps        

def remove_files(root,path_list,counter):
    """
    Removes files from the path_list that are given in the txt file, related to read_out_txt_file
    """
    files_to_be_skipped = read_out_txt_file(root)
    counter_local = 0
    path_list_copy = path_list.copy()
    for file in path_list_copy:
        date_of_file = file.split('/')[-1].split('.')[0].split('_')[-1]
        for timestamp in files_to_be_skipped:
            if date_of_file == timestamp:
                try:
                    path_list.remove(file)
                    counter += 1
                    counter_local += 1
                except:
                    print(f"File {file} not in list")
                    pass
                
    print(f"Removed {counter_local} files of {len(files_to_be_skipped)} files given by the txt file")
    return counter


def plot_only_map(wokring_directory):
    # Below is code for only plotting the map with the defined areas
    plot = plotting.ScenarioPlot(wokring_directory, measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=True, add_validation_gates=False, add_track_indexes=False, gamma=3.5, filename="", dir_name="", resolution=400)
   
    # rectangleA = RectangleA()
    # rectangleB = RectangleB()
    # rectangleC = RectangleC()
    # rectangleD = RectangleD()
    # rectangleE = RectangleE()
    # rectangleF = RectangleF()
    # rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF] 

    plotting.plot_only_map(wokring_directory,radar_circle=True)