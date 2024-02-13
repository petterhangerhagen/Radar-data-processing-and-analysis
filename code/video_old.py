import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
from tqdm import tqdm
import progressbar

from matplotlib.cm import get_cmap
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry import Polygon
from descartes import PolygonPatch
from images_to_video import images_to_video_opencv, empty_folder
import yaml
import os
import datetime
from parameters import tracker_params, measurement_params, process_params, tracker_state
from scipy.io import savemat

# define font size, and size of plots
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.figsize'] = 7.16666, 7.166666

class Video(object):
    """
    A class representing a plot depicitng the tracking scenario.
    """
    def __init__(self, add_covariance_ellipses=False, gamma=3.5,filename="coord_69",dir_name="test",resolution=600,fps=1):
        self.add_covariance_ellipses = add_covariance_ellipses
        self.gamma = gamma
        self.resolution = resolution
        self.filename = filename.split("/")[-1].split("_")[-1].split(".")[0]
        self.dir_name = dir_name
        self.fps = fps
        self.fig, self.ax = plt.subplots()

    def create_video(self, measurements, track_history, ownship, timestamps, ground_truth=None):
        
        # Progress bar
        bar = progressbar.ProgressBar(maxval=len(measurements)).start()

        # Creating a dot where the radar is placed
        self.ax.scatter(0,0,c="black",zorder=10)
        self.ax.annotate(f"Radar",(2,2),zorder=10)

        # Find the limits of the plot
        N_min, N_max, E_min, E_max = find_track_limits(track_history)
        self.ax.set_xlim(E_min, E_max)
        self.ax.set_ylim(N_min, N_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East [m]')
        self.ax.set_ylabel('North [m]')

        # Reads out the measurements and tracks, and get them on the right format
        measurement_dict, track_dict = self.create_dict(measurements, track_history, ownship, timestamps, ground_truth=None)
        measurement_dict.pop("Info")
        track_dict.pop("Info")
        for i,timestamp in enumerate(timestamps):
            # plot measurements
            if timestamp[0] in measurement_dict:
                self.ax.plot(measurement_dict[timestamp[0]][0], measurement_dict[timestamp[0]][1], marker='o', color=measurement_dict[timestamp[0]][2], markersize=3)
            
            # skip first iteration, since we have no track to plot, since we need two points to plot a line
            if i==0:
                last_position = track_dict[timestamp[0]][1:3]
                track_index = track_dict[timestamp[0]][0]
                continue

            # Plotting the track
            if timestamp[0] in track_dict:
                # check if we have a new track
                position = track_dict[timestamp[0]][1:3]
                if track_index != track_dict[timestamp[0]][0]:
                    last_position = track_dict[timestamp[0]][1:3]
                    track_index = track_dict[timestamp[0]][0]
                    # Plotting the last position of the previous track
                    index = list(track_dict.keys()).index(timestamp[0])-1
                    self.ax.plot(track_dict[list(track_dict.keys())[index]][1], track_dict[list(track_dict.keys())[index]][2], 'o', color=track_dict[list(track_dict.keys())[index]][4], markersize=5)
                    continue

                self.ax.plot([last_position[0],position[0]], [last_position[1],position[1]], color=track_dict[timestamp[0]][4], lw=1,ls="-")
                edgecolor = matplotlib.colors.colorConverter.to_rgba(track_dict[timestamp[0]][4], alpha=0)
                facecolor = matplotlib.colors.colorConverter.to_rgba(track_dict[timestamp[0]][4], alpha=0.16)
                covariance_ellipse = get_ellipse(track_dict[timestamp[0]][1:3], track_dict[timestamp[0]][3], gamma=self.gamma)
                self.ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))
                last_position = track_dict[timestamp[0]][1:3]
                if i == len(timestamps)-2:
                    self.ax.plot(last_position[0], last_position[1], 'o', color=track_dict[timestamp[0]][4], markersize=5)

            # Saving the frame
            self.ax.set_title(f"Time: {timestamp[0]}")
            self.fig.savefig(f'/home/aflaptop/Documents/data_mradmin/tracking_results/videos/temp/tracker_{i+1}.png',dpi=100)
            bar.update(i)
   
        # Saving the video
        photos_file_path = "/home/aflaptop/Documents/data_mradmin/tracking_results/videos/temp"
        video_name = f'{photos_file_path[:-4]}{self.filename}.avi'
        images_to_video_opencv(photos_file_path, video_name, self.fps)
        print(f"Saving {video_name.split('/')[-1]}")
        empty_folder(photos_file_path)

    def create_dict(self, measurements, track_history, ownship, timestamps, ground_truth=None):
        # Reads out the measurements and save them to a directory
        cmap = get_cmap('Greys')
        measurements_all = measurements
        measurements_all = dict((i, set(measurements)) for i, measurements in enumerate(measurements_all))
        timestamps = np.asarray(timestamps)
        interval = (timestamps-timestamps[0]+timestamps[-1]/5)/(timestamps[-1]-timestamps[0]+timestamps[-1]/5)
        measurement_dict = {}
        measurement_dict["Info"] = ["x","y","color"]
        for index, measurement_set in measurements_all.items():
            measurement_color = cmap(interval[index].squeeze())
            for measurement in measurement_set:
                if measurement.timestamp not in measurement_dict:
                    measurement_dict[measurement.timestamp] = [measurement.value[0], measurement.value[1], measurement_color]
                else:
                    measurement_dict[measurement.timestamp].append([measurement.value[0], measurement.value[1], measurement_color])

        # Reads out the tracks and save them to a directory
        color_idx = 0
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0'] # Orange, blå, grønn, rød, rosa
        color = None
        track_dict = {}
        track_dict["Info"] = ["Track index","x","y","covariance","color"]
        for index, trajectory in track_history.items():
            # Assigning a color to the track
            if color is not None:
                selected_color = color
            else:
                selected_color = colors[color_idx%5] 
                color_idx += 1
            # Adding the track to the dictionary
            for track in trajectory:
                if track.timestamp not in track_dict:
                    track_dict[track.timestamp] = [index, track.posterior[0][0], track.posterior[0][2], track.posterior[1][0:3:2,0:3:2], selected_color]
                else:
                    track_dict[track.timestamp].append([index, track.posterior[0][0],track.posterior[0][2], track.posterior[1][0:3:2,0:3:2], selected_color])

        # Saving the dictionaries
        np.save("/home/aflaptop/Documents/radar_tracker/data/track_dict.npy",track_dict)
        np.save("/home/aflaptop/Documents/radar_tracker/data/measurement_dict.npy",measurement_dict)
        save_track_to_mat(track_dict, "/home/aflaptop/Documents/radar_tracker/data/track_dict.mat")
        save_measurement_to_mat(measurement_dict, "/home/aflaptop/Documents/radar_tracker/data/measurement_dict.mat")
        return measurement_dict, track_dict
    
def save_track_to_mat(data_dict, filename):
    matlab_data = {}
    for key, value in data_dict.items():
        # Convert the key to a string
        str_key = str(key)
        matlab_data[str_key] = {
            'Track index': value[0],
            'x': value[1],
            'y': value[2],
            'covariance': value[3],
            'color_code': value[4]
        }
    # Save the data to the MATLAB file
    savemat(filename, matlab_data)
        
def save_measurement_to_mat(data_dict, filename):
    matlab_data = {}
    for key, value in data_dict.items():
        # Convert the key to a string
        str_key = str(key)
        matlab_data[str_key] = {
            'x': value[0],
            'y': value[1],
            'color:code': value[2]
        }
    # Save the data to the MATLAB file
    savemat(filename, matlab_data)

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

def find_track_limits(track_history, extra_spacing=50):
    N_min, N_max, E_min, E_max = np.inf, -np.inf, np.inf, -np.inf
    for track_id, trajectory in track_history.items():
        for track in trajectory:
            mean = track.posterior[0]
            if mean[2] < N_min:
                N_min = mean[2]
            if mean[2] > N_max:
                N_max = mean[2]
            if mean[0] < E_min:
                E_min = mean[0]
            if mean[0] > E_max:
                E_max = mean[0]
    N_min -= extra_spacing
    N_max += extra_spacing
    E_min -= extra_spacing
    E_max += extra_spacing
    return N_min, N_max, E_min, E_max

