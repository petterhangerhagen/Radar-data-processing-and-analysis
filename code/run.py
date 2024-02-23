import import_radar_data
import plotting
import video
import numpy as np
import os
import glob
import datetime

import utilities.merged_measurements.merged_measurement as merged_measurement
from utilities.multi_target.multi_target_scenarios import multi_target_scenarios, move_plot_to_this_directory
from utilities.multi_path.multi_path import check_for_multi_path
from utilities.check_start_and_stop import CountMatrix
from utilities.check_start_and_stop import RectangleA, RectangleB, RectangleC, RectangleD, RectangleE, RectangleF
import utilities.utilities as util

from parameters import tracker_params, measurement_params, process_params, tracker_state
from tracking import constructs, utilities, filters, models, initiators, terminators, managers, associators, trackers


#username = os.getenv('USERNAME') if os.name == 'nt' else os.getenv('USER')

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
    """
    All tracker parameters are imported from parameters.py, and can be changed
    there.
    """
    # make new directory with the current date to save results
    dir_name = util.make_new_directory()

    # turn off tracker functionality
    IMM_off = tracker_state["IMM_off"]
    single_target = tracker_state["single_target"]
    visibility_off = tracker_state["visibility_off"]

    # turn on/off different functionalities
    # 1: True, 0: False
    plot_statement = 0
    relative_to_map = 0
    video_statement = 0
    remove_track_with_low_coherence_factor = 1
    check_for_multi_target_scenarios = 0
    check_for_merged_measurements = 0
    check_for_multi_path_scenarios = 1
    counting_matrix = 0
    reset_count_matrix = 0
    
    # Define count matrix
    if counting_matrix:
        count_matrix = CountMatrix(reset=reset_count_matrix)

    ### Import data ###
    import_selection = 4

    ## All data
    if import_selection == 0:
        root = "/home/aflaptop/Documents/radar_data/"
        path_list = glob.glob(os.path.join(root,'**' ,'*.json'))

    ## Specific data
    elif import_selection == 1:
        # root = "/home/aflaptop/Documents/radar_data/data_aug_15-18"
        root = "/home/aflaptop/Documents/radar_data/data_aug_18-19"
        # root = "/home/aflaptop/Documents/radar_data/data_aug_22-23"
        # root = "/home/aflaptop/Documents/radar_data/data_aug_25-26-27"
        # root = "/home/aflaptop/Documents/radar_data/data_aug_28-29-30-31"
        # root = "/home/aflaptop/Documents/radar_data/data_sep_1-2-3-4-5-6-7"
        # root = "/home/aflaptop/Documents/radar_data/data_sep_8-9-11-14"
        # root = "/home/aflaptop/Documents/radar_data/data_sep_17-18-19-24"
        path_list = glob.glob(os.path.join(root, '*.json'))
    

    # Multi target scenarios
    elif import_selection == 2:
        root = "/home/aflaptop/Documents/radar_data/"
        txt_filename = "/home/aflaptop/Documents/radar_tracker/code/utilities/multi_target/multi_target_scenarios.txt"
        path_list = util.find_files(root,txt_filename)
        

    # Merged measurements
    elif import_selection == 3:
        root = "/home/aflaptop/Documents/radar_data/"
        txt_filename = "/home/aflaptop/Documents/radar_tracker/code/utilities/merged_measurements/merged_measurements.txt"
        path_list = util.find_files(root,txt_filename)
        

    # Multi path scenarios
    elif import_selection == 4:
        root = "/home/aflaptop/Documents/radar_data/"
        txt_filename = "/home/aflaptop/Documents/radar_tracker/code/utilities/multi_path/multi_path_scenarios.txt"
        path_list = util.find_files(root,txt_filename)


    # Specific file
    elif import_selection == 5:
        path_list = ["/home/aflaptop/Documents/radar_data/data_sep_17-18-19-24/rosbag_2023-09-17-15-57-01.json"]
    
    # Empty list
    else: 
        path_list = []

    number_of_multiple_target_scenarios = 0
    number_of_merged_measurements_scenarios = 0
    number_of_multi_path_scenarios = 0
    for i,filename in enumerate(path_list):
        if True:
            print(f'File number {i+1} of {len(path_list)}')
            print(f"Curent file: {os.path.basename(filename)}\n")

            # read out data
            measurements, ownship, timestamps = import_radar_data.radar_data_json_file(filename,relative_to_map=relative_to_map)

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
            unvalid_tracks = util.check_coherence_factor(manager.track_history,coherence_factor=0.75)

            # Check speed of tracks
            unvalid_tracks, track_lengths_dict = util.check_speed_of_tracks(unvalid_tracks, manager.track_history)

            # check for to short tracks
            unvalid_tracks = util.check_lenght_of_tracks(unvalid_tracks, track_lengths_dict)
                    
            # Print current tracks
            util.print_current_tracks(manager.track_history)

            # Remove unvalid tracks
            if remove_track_with_low_coherence_factor:
                for track in unvalid_tracks:
                    del manager.track_history[track]
            
                # Print current tracks
                print("After removing tracks with low coherence factor")
                util.print_current_tracks(manager.track_history)

            # Check for merged measurements
            if check_for_merged_measurements and remove_track_with_low_coherence_factor and not relative_to_map:
                measurement_dict, track_dict = merged_measurement.create_dict(filename, manager.track_history)

                if merged_measurement.merged_measurements(filename, manager.track_history, plot_scenarios=True, return_true_or_false=True):
                    number_of_merged_measurements_scenarios += 1
                    with open("/home/aflaptop/Documents/radar_tracker/code/utilities/merged_measurements/merged_measurements.txt", "a") as f:
                       f.write(os.path.basename(filename) + "\n")
                print(f"Number of merged measurement scenarios: {number_of_merged_measurements_scenarios}\n")
    

            # Video vs image
            if plot_statement:
                # plotting
                if relative_to_map:
                    plot = plotting.ScenarioPlot(measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=True, add_validation_gates=False, add_track_indexes=False, gamma=3.5, filename=filename, dir_name=dir_name, resolution=400)
                    plot.create_with_map(measurements, manager.track_history, ownship, timestamps)
                else:
                    plot = plotting.ScenarioPlot(measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=True, add_validation_gates=False, add_track_indexes=False, gamma=3.5, filename=filename, dir_name=dir_name, resolution=400)
                    plot.create(measurements, manager.track_history, ownship, timestamps)
            
                # rectangleA = RectangleA()
                # rectangleB = RectangleB()
                # rectangleC = RectangleC()
                # rectangleD = RectangleD()
                # rectangleE = RectangleE()
                # rectangleF = RectangleF()
                # rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF] 
                # plotting.plot_only_map(rectangles)
                
            if video_statement:
                inp = input("Do you want to create a video? (y/n): ")
                if inp == "y":
                    video_manager = video.Video(add_covariance_ellipses=True, gamma=1, filename=filename,resolution=100,fps=1)
                    video_manager.create_video(measurements, manager.track_history, ownship, timestamps)
                else:
                    print("No video created")

            if counting_matrix:
                # Check start and stop of tracks
                count_matrix.check_start_and_stop(track_history=manager.track_history,filename=filename)


             # Check for multi-target scenarios
            if check_for_multi_target_scenarios and remove_track_with_low_coherence_factor and not relative_to_map:
                if multi_target_scenarios(manager.track_history):
                    number_of_multiple_target_scenarios += 1
                    with open("/home/aflaptop/Documents/radar_tracker/code/utilities/multi_target/multi_target_scenarios.txt", "a") as f:
                        f.write(os.path.basename(filename) + "\n")
                    if plot_statement:
                        move_plot_to_this_directory(filename, dir_name)
                    #print(f"Number of multiple target scenarios: {number_of_multiple_target_scenarios}\n")

                print(f"Number of multi-target scenarios: {number_of_multiple_target_scenarios}\n")

            # if i==0:
            #     util.histogram_of_tracks_duration(manager.track_history,reset=True)
            # else:
            #     util.histogram_of_tracks_duration(manager.track_history,reset=False)
            
            if check_for_multi_path_scenarios:
                merged_measurement.create_dict(filename, manager.track_history)

                if not plot_statement:
                    plot = None

                if check_for_multi_path(filename, plot, measurements, manager.track_history, timestamps, plot_statement):
                    number_of_multi_path_scenarios += 1
                    print("Multi path scenario found")
                    txt_filename = "/home/aflaptop/Documents/radar_tracker/code/utilities/multi_path/multi_path_scenarios.txt"
                    util.write_filenames_to_txt(filename, txt_filename)
                    # with open("/home/aflaptop/Documents/radar_tracker/code/utilities/multi_path/multi_path_scenarios.txt", "a") as f:
                    #     f.write(os.path.basename(filename) + "\n")
                    
    #util.plot_histogram_of_tracks_duration()
    print(f"Number of multi-path scenarios: {number_of_multi_path_scenarios}\n")
    print("End of run.py")
    if counting_matrix:
        unvalidated_tracks = {"Number of tracks": count_matrix.number_of_tracks,"Unvalidated tracks": count_matrix.unvalidated_track}
        print(f"unvalidated tracks: {count_matrix.unvalidated_track}\n")
        np.save("/home/aflaptop/Documents/radar_tracker/code/npy_files/unvalidated_tracks.npy",unvalidated_tracks)
        files_with_tracks_on_diagonal = {"Number of tracks on diagonal":count_matrix.number_of_tracks_on_diagonal,"Files":count_matrix.files_with_tracks_on_diagonal}
        np.save("/home/aflaptop/Documents/radar_tracker/code/npy_files/files_with_track_on_diagonal.npy",files_with_tracks_on_diagonal)
        count_matrix.track_average_length()
        print(f"average length matrix: {count_matrix.average_length_matrix}\n")
    