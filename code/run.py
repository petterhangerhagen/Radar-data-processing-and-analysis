from tracking import constructs, utilities, filters, models, initiators, terminators, managers, associators, trackers
from parameters import tracker_params, measurement_params, process_params, tracker_state
from check_start_and_stop import CountMatrix

import import_data
import import_radar_data
import plotting
import numpy as np
import os
import glob
import datetime

def setup_manager():
    if IMM_off:
        kinematic_models = [models.CVModel(process_params['cov_CV_high'])]
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
    root = "/home/aflaptop/Documents/data_mradmin/tracking_results"
    todays_date = datetime.datetime.now().strftime("%d-%b")
    path = os.path.join(root,todays_date)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

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
    video = False

    # Define count matrix
    count_matrix = CountMatrix(reset=True)

    all_data = True
    # Import data
    if all_data:
        root1 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_aug_15-18"
        root2 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_aug_18-19"
        root3 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_aug_22-23"
        root4 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_aug_25-26-27"
        root5 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_aug_28-29-30-31"
        root6 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_sep_1-2-3-4-5-6-7"
        root7 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_sep_8-9-11-14"
        root8 = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_sep_17-18-19-24"
        root_list = [root1,root2,root3,root4,root5,root6,root7,root8]
        path_list = []
        for root in root_list:
            temp = glob.glob(os.path.join(root, '*.mat'))
            path_list.extend(temp)
    else:
        path_list = ["/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_sep_8-9-11-14/rosbag_2023-09-08-18-10-43.mat"]

    # root = "/home/aflaptop/Documents/data_mradmin/processed_data/rosbag_markerArray_data_sep_8-9-11-14"
    # temp = glob.glob(os.path.join(root, '*.mat'))
    # path_list.extend(temp)

    for i,filename in enumerate(path_list):
        if i<10:
            print(f'File number {i+1} of {len(path_list)}')
            # read out data
            measurements, ownship, timestamps = import_radar_data.radar_data_mat_file(filename)

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


            # plotting
            plot = plotting.ScenarioPlot(
                measurement_marker_size=3,
                track_marker_size=5,
                add_covariance_ellipses=True,
                add_validation_gates=False,
                add_track_indexes=False,
                gamma=3.5,
                filename=filename,
                dir_name=dir_name,
                resolution=100
            )
            # Video vs image
            # if not video:
            #     plot.create(measurements, manager.track_history, ownship, timestamps)
            # if video:
            #     plot.create_video(measurements, manager.track_history, ownship, timestamps)

            # Check start and stop of tracks
            count_matrix.check_start_and_stop(track_history=manager.track_history,filename=filename)

    unvalidated_tracks = {"Number of tracks": count_matrix.number_of_tracks,"Unvalidated tracks": count_matrix.unvalidated_track}
    np.save("/home/aflaptop/Documents/radar_tracker/data/unvalidated_tracks.npy",unvalidated_tracks)
    files_with_tracks_on_diagonal = {"Number of tracks on diagonal":count_matrix.number_of_tracks_on_diagonal,"Files":count_matrix.files_with_tracks_on_diagonal}
    np.save("/home/aflaptop/Documents/radar_tracker/data/files_with_track_on_diagonal.npy",files_with_tracks_on_diagonal)