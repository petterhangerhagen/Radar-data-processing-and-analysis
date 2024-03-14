"""
Script Title: run.py
Author: Petter Hangerhagen and Audun Gullikstad Hem
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script is part of Audun Gullikstad Hem mulit-target tracker (https://doi.org/10.24433/CO.3351829.v1). It is used as foundation which the code is built on.
This script is used to run the multi-target tracker, and to check for multi-target scenarios.
It also filters out invalid tracks, based on coherence factor, minimum number of states, average speed of tracks and jumping tracks.
The script can also be used to create videos and plots of the scenarios.
"""

import import_radar_data
import plotting
import video
import numpy as np
import os
import glob
import warnings

from utilities.multi_target.multi_target import multi_target_scenarios, move_plot_to_this_directory
from utilities.check_start_and_stop import CountMatrix
import utilities.utilities as util

from parameters import tracker_params, measurement_params, process_params, tracker_state
from tracking import constructs, utilities, filters, models, initiators, terminators, managers, associators, trackers
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")


"""
IMPORTANT: Need to change the radar_data_path and wokring_directory to the correct paths!!
"""
# radar_data_path = "/home/petter/radar_data"
# wokring_directory = "/home/petter/Radar-data-processing-and-analysis"
radar_data_path = "/home/aflaptop/Documents/radar_data"
wokring_directory = "/home/aflaptop/Documents/radar_tracker"

# setup the tracker based on the parameters in parameters.py
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
    
if __name__ == '__main__':
    # Make results directory
    util.make_results_directory(wokring_directory)

    # make new directory with the current date to save results
    dir_name = util.make_new_directory(wokring_directory)

    # turn off tracker functionality
    IMM_off = tracker_state["IMM_off"]
    single_target = tracker_state["single_target"]
    visibility_off = tracker_state["visibility_off"]

    """
    Below are the different statements that can be used to control different functionalities of the code.
    plot_statement: If True, the code will plot the scenario, and save the plot in the radar_tracking_results directory. This directory will be located in the same directory as the you cloned this repo.
    relative_to_map: If True, the code will plot the scenario relative to the map. This will use the occupancy_grid.npy file to plot the map in the background. This can not be used for video.
    video_statement: If True, the code will create a video of the scenario. This is useful for debugging and for visualizing the resutls.
    filter_out_unvalid_tracks: If True, the code will filter out unvalid tracks. This includes tracks with low coherence factor, low speed, and short duration.
    check_for_multi_target_scenarios: If True, the code will check for multi target scenarios. The scenarios will be saved in a txt file in the utilities/multi_target directory. 
    count_and_plot_histogram_of_tracks_duration: If True, the code will count the duration of the tracks, and plot a histogram of the duration.
    counting_matrix: If True, the code will create a counting matrix, which defines the traffic matrix. The defined areas can be seen in the code/utilities/how_areas_are_defined_on_map.jpg. The matrix will be saved in the code/npy_files directory.
    reset_count_matrix: If True, the code will reset the counting matrix. This is useful if you want to start from scratch.
    """
    # 1: True, 0: False
    plot_statement = 1
    relative_to_map = 1
    video_statement = 0
    filter_out_unvalid_tracks = 1
    check_for_multi_target_scenarios = 1
    count_and_plot_histogram_of_tracks_duration = 0
    counting_matrix = 0
    reset_count_matrix = 0
    
    # Define count matrix
    if counting_matrix:
        count_matrix = CountMatrix(wokring_directory, reset=reset_count_matrix)


    """
    The import_selection variable is used to import different data. The different options are:
    0: All data
    1: Specific data, for example only one of the sets of data.
    2: Multi-target scenarios, this will import the data from the multi_target_scenarios.txt file.
    3: Single scenario, hardcode the path to the scenario you want to import.
    """
  

    def scenario_selector(import_selection):
        ## All data
        if import_selection == 0:
            root = radar_data_path
            path_list = glob.glob(os.path.join(root,'**' ,'*.json'))

        ## Specific data
        elif import_selection == 1:
            root = f"{radar_data_path}/data_aug_15-18"
            # root = f"{radar_data_path}/data_aug_18-19"
            # root = f"{radar_data_path}/data_aug_22-23"
            # root = f"{radar_data_path}/data_aug_25-26-27"
            # root = f"{radar_data_path}/data_aug_28-29-30-31"
            # root = f"{radar_data_path}/data_sep_1-2-3-4-5-6-7"
            # root = f"{radar_data_path}/data_sep_8-9-11-14"
            # root = f"{radar_data_path}/data_sep_17-18-19-24"
            path_list = glob.glob(os.path.join(root, '*.json'))
        

        # Multi target scenarios
        elif import_selection == 2:
            root = radar_data_path
            txt_filename = f"{wokring_directory}/code/utilities/multi_target/multi_target_scenarios.txt"
            path_list = util.find_files(root,txt_filename)

        # Single scenario
        elif import_selection == 3:
            root = radar_data_path
            # Both multi-target scenarios in the paper
            path_list = [f"{root}/data_aug_18-19/rosbag_2023-08-19-17-42-41.json",
                         f"{root}/data_aug_18-19/rosbag_2023-08-19-11-21-26.json",]
        
        # Empty list
        else: 
            path_list = []
        return path_list
    
    ### Import data ###
    import_selection = 3
    path_list = scenario_selector(import_selection)
    ###################

    # Plot map with rectangles
    plot_rectangle_map = False
    if plot_rectangle_map:
        util.plot_map_with_rectangles(wokring_directory)

    # Plot only map
    plot_only_map = False
    if plot_only_map:
        util.plot_only_map(wokring_directory)

    # Counting the number of different scenarios
    number_of_multi_target_scenarios = 0
    for i,filename in enumerate(path_list):
        # If statement is used if not all of the imported data should be used
        # for example if only the first 10 files should be used, if i<10
        if False:
            print(f'File number {i+1} of {len(path_list)}')
            print(f"Curent file: {os.path.basename(filename)}\n")

            # read out data
            measurements, ownship, timestamps = import_radar_data.radar_data_json_file(filename)

            # Check i there are any measurements in the file
            if len(measurements) == 0:
                print("No measurements in file")
                continue

            # Check time gap between measurements
            util.check_timegaps(timestamps)

            # define tracker evironment
            manager = setup_manager()
            
            # run tracker
            for k, (measurement_set, timestamp, radar) in enumerate(zip(measurements, timestamps, *ownship.values())):
                #print(f'Timestep {k}:')
                manager.step(measurement_set, float(timestamp), ownship=radar)
                #print(f'Active tracks: {np.sort([track.index for track in manager.tracks])}\n')

            
            # Calculate coherence factor 
            invalid_tracks = util.check_coherence_factor(manager.track_history,coherence_factor=0.75)

            # Check speed of tracks
            invalid_tracks = util.check_invalid_tracks(invalid_tracks, manager.track_history)

            # Count the number of filtered out invalid tracks
            if i == 0:
                util.count_filtered_out_invalid_tracks(wokring_directory,invalid_tracks, reset = True)
            else:
                util.count_filtered_out_invalid_tracks(wokring_directory,invalid_tracks)

            # Remove unvalid tracks
            invalid_track_history = {}
            if filter_out_unvalid_tracks:
                for track in invalid_tracks:
                    invalid_track_history[track] = manager.track_history[track]
                    del manager.track_history[track]
            
            # Plot of the scenario
            if plot_statement:
                # Check if the scenario should be plotted relative to the map
                if relative_to_map:
                    plot = plotting.ScenarioPlot(wokring_directory, measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=True, add_validation_gates=False, add_track_indexes=False, gamma=3.5, filename=filename, dir_name=dir_name, resolution=400)
                    plot.create_with_map(measurements, manager.track_history, invalid_track_history, timestamps)
                else:
                    plot = plotting.ScenarioPlot(wokring_directory, measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=True, add_validation_gates=False, add_track_indexes=False, gamma=3.5, filename=filename, dir_name=dir_name, resolution=400)
                    ax = plot.create(measurements, manager.track_history, invalid_track_history, timestamps)


            # Create video
            if video_statement and not relative_to_map:
                inp = input("Do you want to create a video? (y/n): ")
                if inp == "y":
                    video_manager = video.Video(wokring_directory, add_covariance_ellipses=True, gamma=1, filename=filename,resolution=100,fps=1)
                    video_manager.create_video(measurements, manager.track_history, ownship, timestamps)
                else:
                    print("No video created")

            # Check start and stop of tracks with respect to the defined areas
            if counting_matrix:
                count_matrix.check_start_and_stop(track_history=manager.track_history,filename=filename)


            # Check for multi-target scenarios
            # Can not be used with relative_to_map and without filter_out_unvalid_tracks
            if check_for_multi_target_scenarios and filter_out_unvalid_tracks:
                if multi_target_scenarios(manager.track_history):
                    number_of_multi_target_scenarios += 1
                    txt_filename = f"{wokring_directory}/code/utilities/multi_target/multi_target_scenarios.txt"
                    util.write_filenames_to_txt(filename, txt_filename)
                if plot_statement:
                    move_plot_to_this_directory(wokring_directory, filename, dir_name)
                    
            # Count the duration of the tracks
            if count_and_plot_histogram_of_tracks_duration:
                npy_file_1 = f"{wokring_directory}/code/npy_files/track_duration_1.npy"
                npy_file_2 = f"{wokring_directory}/code/npy_files/track_duration_2.npy"
                combined_dict = {}
                for key in manager.track_history:
                    combined_dict[key] = manager.track_history[key]
                for key in invalid_track_history:
                    combined_dict[key] = invalid_track_history[key]
                if i==0:
                    print("Reset")
                    util.histogram_of_tracks_duration(npy_file_1, manager.track_history, reset=True)
                    util.histogram_of_tracks_duration(npy_file_2, combined_dict, reset=True)
                else:
                    util.histogram_of_tracks_duration(npy_file_1, manager.track_history, reset=False)
                    util.histogram_of_tracks_duration(npy_file_2, combined_dict, reset=False)
    # End of for loop
                          
    # Plot histogram of tracks duration
    if count_and_plot_histogram_of_tracks_duration:
        npy_file_1 = f"{wokring_directory}/code/npy_files/track_duration_1.npy"
        npy_file_2 = f"{wokring_directory}/code/npy_files/track_duration_2.npy"
        util.plot_histogram_of_tracks_duration(npy_file_1,wokring_directory,num=1)
        util.plot_histogram_of_tracks_duration(npy_file_2,wokring_directory,num=2)


    if check_for_multi_target_scenarios:
        print(f"Number of multi-target scenarios: {number_of_multi_target_scenarios}\n")

    if counting_matrix:
        # Save unvalidated tracks
        unvalidated_tracks = {"Number of tracks": count_matrix.number_of_tracks,"Unvalidated tracks": count_matrix.unvalidated_track}
        np.save(f"{wokring_directory}/code/npy_files/unvalidated_tracks.npy",unvalidated_tracks)

        # Save files with tracks on diagonal
        files_with_tracks_on_diagonal = {"Number of tracks on diagonal":count_matrix.number_of_tracks_on_diagonal,"Files":count_matrix.files_with_tracks_on_diagonal}
        np.save(f"{wokring_directory}/code/npy_files/files_with_track_on_diagonal.npy",files_with_tracks_on_diagonal)

        # Compute average length of tracks
        count_matrix.track_average_length()
    
    print("End of run.py")