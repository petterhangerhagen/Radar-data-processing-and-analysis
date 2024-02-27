"""
Script Title: Multi Target
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script contains the classes and functions that are used to check for multi target in the radar tracking pipeline.
It checks for tracks that are close to each other, and then checks if the mahalanobis distance is less than a threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry import Polygon
from descartes import PolygonPatch
import shutil
import os
import re

def multi_target_scenarios(track_history):
    threshold_distance = 20
    n = 3 # scale of covariance matrix

    track_dict = create_dict(track_history)
    track_dict.pop("Info")
    for timestamp, tracks in track_dict.items():
        for i in range(len(tracks)):
            for j in range(i+1,len(tracks)):
                track1, cov1 = [tracks[i][1],tracks[i][2]], tracks[i][3]
                track2, cov2 = [tracks[j][1],tracks[j][2]], tracks[j][3]
                cov1, cov2 = cov1, cov2
                distance = np.sqrt((track1[0]-track2[0])**2 + (track1[1]-track2[1])**2)
                if distance < threshold_distance:
                    cov1_inv = np.linalg.inv(cov1)
                    cov2_inv = np.linalg.inv(cov2)
                    dxy = np.array([track2[0] - track1[0], track2[1] - track1[1]])
                    mahalanobis1 = np.sqrt(np.dot(np.dot(dxy.T, cov1_inv), dxy))
                    mahalanobis2 = np.sqrt(np.dot(np.dot(dxy.T, cov2_inv), dxy))
                    if mahalanobis1 <= n or mahalanobis2 <= n:
                        return True
    return False

def move_plot_to_this_directory(wokring_directory, filename, dirname):
    source_dir = dirname
    destaination_dir = f"{wokring_directory}/code/utilities/multi_target/multi_target_plots"
    partial_file_name = os.path.basename(filename)
    partial_file_name = partial_file_name.split(".")[0]
    partial_file_name = partial_file_name.split("_")[-1]

    files_in_dir = os.listdir(source_dir)
    # Find the matching file
    matching_files = []

    # Loop through each file in the directory
    for file in files_in_dir:
        # Check if the file matches the pattern
        if re.match(f"{partial_file_name}.*\\.png", file):
            # If it matches, add it to the list of matching files
            matching_files.append(file)
            
    for file in matching_files:
        shutil.copy(os.path.join(source_dir, file), destaination_dir)



def create_dict(track_history):
        track_dict = {}
        track_dict["Info"] = ["Track index","x","y","covariance"]
        for index, trajectory in track_history.items():
            for track in trajectory:
                if track.timestamp not in track_dict:
                    track_dict[track.timestamp] = [[index, track.posterior[0][0], track.posterior[0][2], track.posterior[1][0:3:2,0:3:2]]]
                else:
                    track_dict[track.timestamp].append([index, track.posterior[0][0],track.posterior[0][2], track.posterior[1][0:3:2,0:3:2]])

        return track_dict

def get_ellipse(center, Sigma, gamma=1):
    """
    Returns an ellipse. For a covariance ellipse, gamma is the square of the
    number of standard deviations from the mean.
    Method from https://cookierobotics.com/007/.
    """
    lambda_, _ = np.linalg.eig(Sigma)
    lambda_root = np.sqrt(lambda_)
    width = lambda_root[0]*np.sqrt(gamma)
    height = lambda_root[1]*np.sqrt(gamma)
    rotation = np.rad2deg(np.arctan2(lambda_[0]-Sigma[0,0], Sigma[0,1]))
    circ = Point(center).buffer(1)
    non_rotated_ellipse = affinity.scale(circ, width, height)
    ellipse = affinity.rotate(non_rotated_ellipse, rotation)
    edge = np.array(ellipse.exterior.coords.xy)
    return Polygon(edge.T)
