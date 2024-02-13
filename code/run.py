from tracking import constructs, utilities, filters, models, initiators, terminators, managers, associators, trackers
from parameters import tracker_params, measurement_params, process_params, tracker_state
from check_start_and_stop import CountMatrix

import import_data
import import_radar_data
import plotting
import video
import numpy as np
import os
import glob
import datetime

def setup_manager():
    if IMM_off:
        kinematic_models = [models.CVModel(process_params['cov_CV_single'])]
        pi_matrix = np.array([[1]])
        init_mode_probs = np.array([1])
    else:
        kinematic_models = [models.CVModel(process_params['cov_CV_low']),models.CTModel(process_params['cov_CV_low'],process_params['cov_CT']),models.CVModel(process_params['cov_CV_high'])]
        pi_matrix = process_params['pi_matrix']
        init_mode_probs = process_params['init_mode_probs']
    clutter_model = models.ConstantClutterModel(tracker_params['clutter_density'])
    
    measurement_model = models.CombinedMeasurementModel(
        measurement_mapping = measurement_params['measurement_mapping'],
        cartesian_covariance = measurement_params['cart_cov'],
        range_covariance = measurement_params['range_cov'],
        bearing_covariance = measurement_params['bearing_cov'])

    filter = filters.IMMFilter(
        measurement_model = measurement_model,
        mode_transition_matrix = pi_matrix)

    data_associator = associators.MurtyDataAssociator(
        n_measurements_murty = 4,
        n_tracks_murty = 2,
        n_hypotheses_murty = 8)

    tracker = trackers.VIMMJIPDATracker(
        filter,
        clutter_model,
        data_associator,
        survival_probability=tracker_params['survival_prob'],
        visibility_transition_matrix = tracker_params['visibility_transition_matrix'],
        detection_probability=tracker_params['P_D'],
        gamma=tracker_params['gamma'],
        single_target=single_target,
        visibility_off=visibility_off)

    track_initiation = initiators.SinglePointInitiator(
        tracker_params['init_prob'],
        measurement_model,
        tracker_params['init_Pvel'],
        mode_probabilities = init_mode_probs,
        kinematic_models = kinematic_models,
        visibility_probability = 0.9)

    track_terminator = terminators.Terminator(
        tracker_params['term_threshold'],
        max_steps_without_measurements = 5,
        fusion_significance_level = 0.01)

    track_manager = managers.Manager(tracker, track_initiation, track_terminator, tracker_params['conf_threshold'])
    return track_manager

def make_new_directory():
    # Making new directory for the results
    root = "/home/aflaptop/Documents/radar_tracker/tracking_results"
    todays_date = datetime.datetime.now().strftime("%d-%b")
    path = os.path.join(root,todays_date)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def read_out_txt_file(root):
    txt_file = glob.glob(os.path.join(root, '*.txt'))
    timestamps = []
    with open(txt_file[0], 'r') as f:
        lines = f.readlines()
        for line in lines:
            timestamps.append(line.split()[0])
    return timestamps        

def remove_files(root,path_list,counter):
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

def save_measurements(measurements,timestamps):
    measurement_dict = {}
    for k, (measurement_set, timestamp) in enumerate(zip(measurements, timestamps)):
        measurement_set = list(measurement_set)

        if measurement_set:
            for measurement in measurement_set:
                if not timestamp[0] in measurement_dict:
                    measurement_dict[timestamp[0]] = [[measurement.mean[0],measurement.mean[1]]]
                else:
                    measurement_dict[timestamp[0]].append([measurement.mean[0],measurement.mean[1]])
        else:
            measurement_dict[timestamp[0]] = []
    np.save("/home/aflaptop/Documents/radar_tracker/data/measurements_after_importing.npy",measurement_dict)
    
def check_lenght_of_tracks(track_history):
    for track in track_history.items():
        first_mean, cov = track[1][0].posterior
        last_mean, cov = track[1][-1].posterior
        first_point = (first_mean[0],first_mean[2])
        last_point = (last_mean[0],last_mean[2])
        lenght = np.sqrt((first_point[0]-last_point[0])**2 + (first_point[1]-last_point[1])**2)
        print(f"lenght of track {track[0]} = {lenght:.2f}") 

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

if __name__ == '__main__':
    """
    All tracker parameters are imported from parameters.py, and can be changed
    there.
    """
    # make new directory with the current date to save results
    dir_name = make_new_directory()

    # turn off tracker functionality
    IMM_off = tracker_state["IMM_off"]
    single_target = tracker_state["single_target"]
    visibility_off = tracker_state["visibility_off"]

    plot_statement = 1
    video_statement = 1
    counting_matrix = False
    remove_track_with_low_coherence_factor = 1
    
    # Define count matrix
    if counting_matrix:
        count_matrix = CountMatrix(reset=True)

    ### Import data ###
    #root = "/home/aflaptop/Documents/data_mradmin/json_files/data_aug_15-18/"
    root = "/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19"
    #root = "/home/aflaptop/Documents/data_mradmin/json_files/data_aug_22-23"
    #root = "/home/aflaptop/Documents/data_mradmin/json_files/data_sep_8-9-11-14"
    path_list = glob.glob(os.path.join(root, '*.json'))

    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_15-18/rosbag_2023-08-15-13-33-24.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_15-18/rosbag_2023-08-18-13-06-49.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-18-13-30-58.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-18-13-32-36.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_sep_8-9-11-14/rosbag_2023-09-09-13-47-42.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_sep_8-9-11-14/rosbag_2023-09-08-17-00-11.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_sep_8-9-11-14/rosbag_2023-09-11-12-03-28.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_sep_8-9-11-14/rosbag_2023-09-09-13-15-23.json"]
    # path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-18-14-57-38.json"]
    #path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-18-16-20-33.json"]
    #path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-19-11-18-46.json"]
    #path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-18-17-18-06.json"]
    path_list = ["/home/aflaptop/Documents/data_mradmin/json_files/data_aug_18-19/rosbag_2023-08-19-16-18-26.json"]
    for i,filename in enumerate(path_list):
        if True:
            print(f'File number {i+1} of {len(path_list)}')
            print(f"Curent file: {os.path.basename(filename)}\n")

            # read out data
            measurements, ownship, timestamps = import_radar_data.radar_data_json_file(filename)
        
            # Check time gap between measurements
            check_timegaps(timestamps)

            # Save measurements
            # save_measurements(measurements,timestamps)
            
            # Check i there are any measurements in the file
            if len(measurements) == 0:
                print("No measurements in file")
                continue


            # define tracker evironment
            manager = setup_manager()
            

            # run tracker
            for k, (measurement_set, timestamp, radar) in enumerate(zip(measurements, timestamps, *ownship.values())):
                #print(f'Timestep {k}:')
                manager.step(measurement_set, float(timestamp), ownship=radar)
                #print(f'Active tracks: {np.sort([track.index for track in manager.tracks])}\n')


            # Check lenght of tracks
            #check_lenght_of_tracks(manager.track_history)
            
            
            # Calculate coherence factor 
            unvalid_tracks = check_coherence_factor(manager.track_history,coherence_factor=0.75)

            # Print current tracks
            print_current_tracks(manager.track_history)

            # Remove unvalid tracks
            if remove_track_with_low_coherence_factor:
                for track in unvalid_tracks:
                    del manager.track_history[track]
            
                # Print current tracks
                print("After removing tracks with low coherence factor")
                print_current_tracks(manager.track_history)

            # Video vs image
            if plot_statement:
                # plotting
                plot = plotting.ScenarioPlot(measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=True, add_validation_gates=False, add_track_indexes=False, gamma=3.5, filename=filename, dir_name=dir_name, resolution=400)
                plot.create(measurements, manager.track_history, ownship, timestamps)
            
            if video_statement:
                inp = input("Do you want to create a video? (y/n): ")
                if inp == "y":
                    video_manager = video.Video(add_covariance_ellipses=True, gamma=1, filename=filename, dir_name=dir_name,resolution=100,fps=1)
                    video_manager.create_video(measurements, manager.track_history, ownship, timestamps)
                else:
                    print("No video created")
            # if counting_matrix:
            #     # Check start and stop of tracks
            #     count_matrix.check_start_and_stop(track_history=manager.track_history,filename=filename)

    # if counting_matrix:
    #     unvalidated_tracks = {"Number of tracks": count_matrix.number_of_tracks,"Unvalidated tracks": count_matrix.unvalidated_track}
    #     np.save("/home/aflaptop/Documents/radar_tracker/data/unvalidated_tracks.npy",unvalidated_tracks)
    #     files_with_tracks_on_diagonal = {"Number of tracks on diagonal":count_matrix.number_of_tracks_on_diagonal,"Files":count_matrix.files_with_tracks_on_diagonal}
    #     np.save("/home/aflaptop/Documents/radar_tracker/data/files_with_track_on_diagonal.npy",files_with_tracks_on_diagonal)