import os
import datetime
import glob
import numpy as np

def make_new_directory():
    # Making new directory for the results
    
    root = "/home/aflaptop/Documents/radar_tracking_results"
    todays_date = datetime.datetime.now().strftime("%d-%b")
    path = os.path.join(root,todays_date)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def check_timegaps(timestamps):
    time_gaps = []
    for k in range(len(timestamps)-1):
        time_gap = timestamps[k+1]-timestamps[k]
        time_gaps.append(time_gap)

    time_gap = np.mean(time_gaps)
    for k in range(len(time_gaps)):
        if  not (0.8*time_gap < time_gaps[k][0] < 1.2*time_gap):
            print(f"Time gap are not consistent, time gap {time_gaps[k][0]:.2f} at index {k}")
    print(f"Time gap between measurements = {time_gap:.2f}\n")

def check_speed_of_tracks(unvalid_tracks, track_history):
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
        print(f"Speed of track {track_id} = {track_speed_knots:.2f} knots")
        if track_speed_knots > 6 and track_id not in unvalid_tracks:
            unvalid_tracks.append(track_id)

    return unvalid_tracks, track_lengths_dict

def check_lenght_of_tracks(unvalid_tracks, track_lengths_dict):
    for track_length in track_lengths_dict.items():
        if track_length[1] < 10 and track_length[0] not in unvalid_tracks:
            unvalid_tracks.append(track_length[0])
    return unvalid_tracks

def check_coherence_factor(track_history,coherence_factor=0.75):
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
        print(f"Coherence factor for track {track[0]} = {ck/len(u):.2f}\n")
        if ck/len(u) < coherence_factor:
            not_valid_tracks.append(track[0])

    print(f"Tracks with to low coherence factor: {not_valid_tracks}\n")
    return not_valid_tracks

def print_current_tracks(track_history):
    print(f"Number of tracks = {len(track_history)}")
    for key, value in track_history.items():
        print(f"Track {key}")
    print("\n")

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