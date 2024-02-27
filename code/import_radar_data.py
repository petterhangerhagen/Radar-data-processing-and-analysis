import scipy.io as io
import numpy as np
from tracking import constructs
from parameters import measurement_params
import json


def radar_data_json_file(wokring_directory, json_file, relative_to_map = False):

    if relative_to_map:
        data = np.load(f"{wokring_directory}/code/npy_files/occupancy_grid.npy",allow_pickle='TRUE').item()
        origin_x = data["origin_y"]
        origin_y = data["origin_x"]

    measurements = []
    timestamps = []

    with open(json_file, 'r') as file:
        data = json.load(file)
    
    timestamp = data[0]["header"]["stamp"]["secs"]
    timestamps_nano = data[0]["header"]["stamp"]["nsecs"]
    timestamp = timestamp + timestamps_nano*10**(-9)
    first_timestamp = timestamp

    for k,item in enumerate(data):

        if k == 0:
           timestamp = 0
        else:
            timestamp = item["header"]["stamp"]["secs"]
            timestamps_nano = item["header"]["stamp"]["nsecs"]
            timestamp = timestamp + timestamps_nano*10**(-9) - first_timestamp
       
        item_data = item["scan"]
        measurements.append(set())
        for i,measurement in enumerate(item_data):
            if relative_to_map:
                y = measurement["cluster_centroid"]["x"] + origin_x
                x = measurement["cluster_centroid"]["y"] + origin_y
            else:
                y = measurement["cluster_centroid"]["x"]
                x = measurement["cluster_centroid"]["y"]
            meas_set = (np.array([x,y]))
            measurements[-1].add(constructs.Measurement(meas_set, measurement_params['cart_cov'],  float(timestamp)))
        timestamps.append(float(timestamp))

    measurements = np.array(measurements)
    timestamps = np.array(timestamps)

    ownship = np.zeros((len(timestamps),5))
    radar = {1: [constructs.State(ownship_pos, np.identity(4), timestamp) for ownship_pos, timestamp in zip(ownship, timestamps)]}

    timestamps = np.reshape(timestamps,(len(timestamps),1))
    return measurements, radar, timestamps