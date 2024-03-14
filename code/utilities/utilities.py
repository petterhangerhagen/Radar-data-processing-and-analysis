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
from shapely.geometry.point import Point
from utilities.check_start_and_stop import RectangleA, RectangleB, RectangleC, RectangleD, RectangleE, RectangleF
import plotting

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

def check_invalid_tracks(unvalid_tracks, track_history):
    """
    If the average speed of a track is higher than 6 knots, the track is considered invalid.+
    """
    # track_lengths_dict = {}
    # for track in track_history.items():
    #     first_mean, cov = track[1][0].posterior
    #     last_mean, cov = track[1][-1].posterior
    #     first_point = (first_mean[0],first_mean[2])
    #     last_point = (last_mean[0],last_mean[2])
    #     track_length = np.sqrt((first_point[0]-last_point[0])**2 + (first_point[1]-last_point[1])**2)
    #     #print(f"lenght of track {track[0]} = {track_length:.2f}") 
    #     track_lengths_dict[track[0]] = track_length

    for track in track_history.items():
        if track[0] in unvalid_tracks:
            continue
        if len(track[1]) <= 20:
            unvalid_tracks.append(track[0])

    max_totale_speed = 5

    track_lengths_dict = {}
    for track in track_history.items():
        if track[0] in unvalid_tracks:
            continue
        track_id = track[0]
        track_start_time = track[1][0].timestamp
        track_end_time = track[1][-1].timestamp
        track_time = track_end_time - track_start_time

        track_lengths = []
        first_mean, cov = track[1][0].posterior
        x_last = first_mean[0]
        y_last = first_mean[2]
        for k, track_point in enumerate(track[1]):
            mean, cov = track_point.posterior
            x = mean[0]
            y = mean[2]
            distance = np.sqrt((x-x_last)**2 + (y-y_last)**2)
            track_lengths.append(distance)
            x_last = x
            y_last = y
        track_lengths_dict[track_id] = sum(track_lengths)


        track_speed = track_lengths_dict[track_id]/track_time
        # print(f"Speed of track {track_id} = {track_speed_knots:.2f} knots")
        if track_speed > max_totale_speed:
            unvalid_tracks.append(track_id)




    speed_of_tracks = {}
    max_speed = 10
    for track in track_history.items():
        if track[0] in unvalid_tracks:
            continue
        speed_of_tracks[track[0]] = []
        last_timestamp = track[1][0].timestamp
        last_mean, cov = track[1][0].posterior
        last_x = last_mean[0]
        last_y = last_mean[2]
        number_of_times_over_max_speed = 0

        for k, track_point in enumerate(track[1]):
            
            if k == 0:  
                continue
            timestamp = track_point.timestamp
            time_gap = timestamp - last_timestamp
            mean, cov = track_point.posterior
            x = mean[0]
            y = mean[2]
            distance = np.sqrt((x-last_x)**2 + (y-last_y)**2)
            speed = distance/time_gap
            speed_of_tracks[track[0]].append(speed)
            if speed > max_speed and track[0] not in unvalid_tracks:
                number_of_times_over_max_speed += 1
                # unvalid_tracks.append(track[0])
            if number_of_times_over_max_speed > 1:
                unvalid_tracks.append(track[0])
                break
            last_x = x
            last_y = y
            last_timestamp = timestamp

    tracks_dict = {}
    for track in track_history.items():
        if track[0] in unvalid_tracks:
            continue
        tracks_dict[track[0]] = []
        for track_point in track[1]:
            mean, cov = track_point.posterior
            tracks_dict[track[0]].append([int(mean[0]),int(mean[2])])

    # for track_id,track_points in tracks_dict.items():
    #     print(f"Track {track_id}")
    #     print(f"Number of points = {len(track_points)}")
    #     print(f"Points = {track_points}")
    #     print("\n")

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

def count_filtered_out_invalid_tracks(working_directory,unvalid_tracks, reset=False):
    """
    Counts the number of unvalid tracks
    """
    filename = f"{working_directory}/code/npy_files/filtered_out_unvalid_tracks.npy"
    if not os.path.exists(filename) or reset:
        number_of_unvalid_tracks = len(unvalid_tracks)
        np.save(filename,number_of_unvalid_tracks)
    else:
        number_of_unvalid_tracks = np.load(filename,allow_pickle=True)
        number_of_unvalid_tracks += len(unvalid_tracks)
        np.save(filename,number_of_unvalid_tracks)
    
def check_if_track_is_stationary(track):
    stationary = True
    if len(track[1]) < 2:
        return True
    first_mean, cov = track[1][0].posterior
    circle = Point(first_mean[0], first_mean[2]).buffer(30)
    for track_point in track[1]:        
        mean, cov = track_point.posterior
        x = mean[0]
        y = mean[2]
        if not circle.contains(Point(x, y)):
            stationary = False
            break
    
    return stationary

def histogram_of_tracks_duration(npy_file, track_history, reset=False):
    """
    Saves the duration of the tracks in a npy file, which later can be used to plot a histogram of the track duration
    """
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

    np.save(npy_file,tracks_duration_dict)

def plot_histogram_of_tracks_duration(npy_file, wokring_directory,num):
    """
    Plots a histogram of the track duration
    """
    tracks_duration_dict = np.load(npy_file,allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(12, 5))
    data1 = [value[0] for value in tracks_duration_dict.values()]
    data2 = [value[1] for value in tracks_duration_dict.values()]
    ax.bar(tracks_duration_dict.keys(), data1, color='#1f77b4', label='Moving tracks')
    ax.bar(tracks_duration_dict.keys(), data2, color='#2ca02c',bottom=data1, label='Stationary tracks')
    ax.set_xlabel('Duration of tracks [s]',fontsize=15)
    ax.set_ylabel('Number of tracks',fontsize=15)
    ax.legend(fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f"{os.path.dirname(wokring_directory)}/radar_tracking_results/histogram_track_duration_{num}.png",dpi=400)
    print(f"Saved histogram of track duration to {os.path.dirname(wokring_directory)}/radar_tracking_results/histogram_track_duration.png")
    plt.close()

def plot_map_with_rectangles(wokring_directory):
    # Below is code for only plotting the map with the defined areas
    rectangleA = RectangleA()
    rectangleB = RectangleB()
    rectangleC = RectangleC()
    rectangleD = RectangleD()
    rectangleE = RectangleE()
    rectangleF = RectangleF()
    rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF] 
    plotting.plot_only_map_with_rectangles(wokring_directory, rectangles)

def plot_only_map(wokring_directory):
    plotting.plot_only_map_with_rectangles(wokring_directory,rectangles=None)
