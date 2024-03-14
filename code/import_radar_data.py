"""
Script Title: Multi Target
Author: Petter Hangerhagen and Audun Gullikstad Hem
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script is inspired form Audun Gullikstad Hem mulit-target tracker (https://doi.org/10.24433/CO.3351829.v1).
It contains a function that reads radar data from a json file and returns the measurements, radar and timestamps. Where the radar position is always zero.
"""

import numpy as np
from tracking import constructs
from parameters import measurement_params
import json


def radar_data_json_file(json_file):

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
