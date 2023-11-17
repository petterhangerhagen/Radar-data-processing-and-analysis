from tracking import constructs, utilities, filters, models, initiators, terminators, managers, associators, trackers
from parameters import tracker_params, measurement_params, process_params, tracker_state

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

def find_csv_files(root):
    # Iterating through the different csv files with detections
    path = os.path.join(root,"data")
    rosbag_list = os.listdir(path)
    path_list = []
    for rosbag_name in rosbag_list:
        csv_name = rosbag_name.split("_")[-1]
        temp_path = os.path.join(path,rosbag_name,f"coordinates_{csv_name}.csv")
        path_list.append(temp_path)
    return path_list

def delete_empty_directories(root):
    # Deleting empty directories
    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            temp_path = os.path.join(root, name)
            if not os.listdir(temp_path):
                os.rmdir(temp_path)

def make_new_directory():
    # Making new directory for the results
    root = "/home/aflaptop/Documents/radar_tracker/results"
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

    # turn off tracker functionality
    IMM_off = tracker_state["IMM_off"]
    single_target = tracker_state["single_target"]
    visibility_off = tracker_state["visibility_off"]
    video = False

    # root = '/home/aflaptop/Documents/data_mradmin/data8-9-11-14'
    # #path_list = find_csv_files(root)

    # path_list = ["/home/aflaptop/Documents/radar_tracker/data/coordinates_2023-09-09-10-37-00.csv",
    #              "/home/aflaptop/Documents/radar_tracker/data/coordinates_2023-09-09-12-02-57.csv",
    #              "/home/aflaptop/Documents/radar_tracker/data/coordinates_2023-09-09-15-08-02.csv",
    #              "/home/aflaptop/Documents/radar_tracker/data/coordinates_2023-09-09-16-16-05.csv"]

    # path_list = ["/home/aflaptop/Documents/radar_tracker/data/bag_2023-10-15-23-14-03.mat",
    #              "/home/aflaptop/Documents/radar_tracker/data/Clustered_data_2023-09-09-10-37-00.mat",
    #              "/home/aflaptop/Documents/radar_tracker/data/Clustered_data_2023-09-09-12-02-57.mat",
    #              "/home/aflaptop/Documents/radar_tracker/data/Clustered_data_2023-09-09-15-08-02.mat",
    #              "/home/aflaptop/Documents/radar_tracker/data/Clustered_data_2023-10-15-14-15-12.mat"]

    # root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_data8-9-11-14"
    # root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_data17-18-19-24"
    # root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_data18-19"
    # root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_data25-26-27"
    # root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_data15-18"
    # root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_data22-23"



    #root = "/home/aflaptop/Documents/data_mradmin/rosbag_markerArray_temp"
    path_list = glob.glob(os.path.join(root, '*.mat'))
    # path_list = [os.path.join(root, "bag_2023-10-15-22-53-53.mat")]
    # path_list = [os.path.join(root, "bag_2023-10-15-23-01-33.mat")]
    # path_list = [os.path.join(root, "bag_2023-10-15-23-01-21.mat")]
    # path_list = [os.path.join(root, "bag_2023-10-15-23-01-55.mat")]
    #path_list = [os.path.join(root, "rosbag_2023-09-01-17-56-56.mat")]
    
   
    dir_name = make_new_directory()

    for i,filename in enumerate(path_list):
        if True:
            #measurements, ownship, timestamps = import_radar_data.radar_data(filename)
            measurements, ownship, timestamps = import_radar_data.radar_data_mat_file(filename)
            if len(measurements) == 0:
                print("No measurements in file")
                continue
            # for (time,measurment) in zip(timestamps,measurements):
            #     print(f"Timestamp: {time}, Measurment: {measurment}")
            # print(measurements[5])
            # measurements, ownship,ais, timestamps = import_data.final_dem()
            
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
                dir_name=dir_name
            )
            if not video:
                plot.create(measurements, manager.track_history, ownship, timestamps)

            if video:
                plot.create_video(measurements, manager.track_history, ownship, timestamps)

            #plot.check_start_and_stop(manager.track_history)
