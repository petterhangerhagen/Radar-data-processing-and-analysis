import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import progressbar
import os

from matplotlib.cm import get_cmap
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry import Polygon
from descartes import PolygonPatch
from utilities.images_to_video import images_to_video_opencv, empty_folder
from parameters import tracker_params, measurement_params, process_params, tracker_state

# define font size, and size of plots
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.figsize'] = 7.16666, 7.166666

class Video(object):
    """
    A class representing a plot depicitng the tracking scenario.
    """
    def __init__(self, add_covariance_ellipses=False, gamma=3.5,filename="coord_69",resolution=100,fps=1):
        self.add_covariance_ellipses = add_covariance_ellipses
        self.gamma = gamma
        self.resolution = resolution
        self.filename = filename.split("/")[-1].split("_")[-1].split(".")[0]
        #self.dir_name = dir_name
        self.fps = fps
        self.fig, self.ax = plt.subplots()

    def create_video(self, measurements, track_history, ownship, timestamps, ground_truth=None):
        
        # Progress bar
        bar = progressbar.ProgressBar(maxval=len(measurements)).start()

        # Creating a dot where the radar is placed
        self.ax.scatter(0,0,c="black",zorder=10)
        self.ax.annotate(f"Radar",(2,2),zorder=10)

        # Find the limits of the plot
        #N_min, N_max, E_min, E_max = find_track_limits(track_history)
        self.ax.set_xlim(-120, 120)
        self.ax.set_ylim(-140, 20)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East [m]')
        self.ax.set_ylabel('North [m]')

        # Reads out the measurements and tracks, and get them on the right format
        measurement_dict, track_dict = self.create_dict(measurements, track_history, ownship, timestamps, ground_truth=None)
        measurement_dict.pop("Info")
        track_dict.pop("Info")
        last_position_dict = {}
        track_started = False
        for i,timestamp in enumerate(timestamps):
            time_stamp = timestamp[0]
            # plot measurements
            if timestamp[0] in measurement_dict:
                for k in range(len(measurement_dict[timestamp[0]])):
                    current_measurement = measurement_dict[timestamp[0]][k]
                    self.ax.plot(current_measurement[0], current_measurement[1], marker='o', color=current_measurement[2], markersize=3)
            
            # skip first iteration, since we have no track to plot, since we need two points to plot a line
            if not track_started:
                if time_stamp in track_dict:
                    for track in track_dict[time_stamp]:
                        track_index = track[0]
                        last_position_dict[track_index] = [track[1:3],track[-1]]
                        position = track[1:3]
                        track_color = track[-1]
                        self.ax.scatter(position[0],position[1],color=track_color,zorder=10)
                        self.ax.annotate(f"Track {track_index}",(track[1],track[2]),zorder=10)
                    track_started = True
            else:
                # Plotting the track
                if time_stamp in track_dict:
                    for k,track in enumerate(track_dict[time_stamp]):
                        # print(track)
                        # print("\n")
                        track_index = track[0]
                        if track_index in last_position_dict.keys():
                            last_position = last_position_dict[track_index][0]
                            track_color = last_position_dict[track_index][1]
                            position = track[1:3]
                            # print(last_position)
                            # print(position)
                            self.ax.plot([last_position[0],position[0]], [last_position[1],position[1]], color=track_color, lw=1,ls="-")

                            edgecolor = matplotlib.colors.colorConverter.to_rgba(track_color, alpha=0)
                            facecolor = matplotlib.colors.colorConverter.to_rgba(track_color, alpha=0.16)
                            covariance_ellipse = get_ellipse(position, track[3], gamma=self.gamma)
                            #print(f"Track {track_index}, covariance: {track[3]}\n")
                            self.ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))


                            last_position_dict[track_index] = [track[1:3],track[-1]]

                        else:
                            position = track[1:3]
                            track_color = track[-1]
                            self.ax.scatter(position[0],position[1],color=track_color,zorder=10)
                            self.ax.annotate(f"Track {track_index}",(track[1],track[2]),zorder=10)
                            last_position_dict[track_index] = [track[1:3],track[-1]]

           
            # Plotting a dot where the tracks stops
            if i>0:
                if timestamps[i-1][0] in track_dict.keys() and timestamps[i][0] in track_dict.keys():
                    previous_track_ids = [track[0] for track in track_dict[timestamps[i-1][0]]]
                    current_track_ids = [track[0] for track in track_dict[timestamps[i][0]]]
                    for previous_track_id in previous_track_ids:
                        if previous_track_id not in current_track_ids:
                            #print("track ended")
                            last_position = last_position_dict[previous_track_id][0]
                            track_color = last_position_dict[previous_track_id][1]
                            self.ax.plot(last_position[0], last_position[1], 'o', color=track_color, markersize=5)
                    
            # Saving the frame
            self.ax.set_title(f"Time: {timestamp[0]:.2f} s")
            #self.fig.savefig(f'/home/aflaptop/Documents/data_mradmin/tracking_results/videos/temp/tracker_{i+1}.png',dpi=self.resolution)
            self.fig.savefig(f'/home/aflaptop/Documents/radar_tracking_results/videos/temp/tracker_{i+1}.png',dpi=self.resolution)

            bar.update(i)
   
        # Saving the video
        #photos_file_path = "/home/aflaptop/Documents/data_mradmin/tracking_results/videos/temp"
        photos_file_path = "/home/aflaptop/Documents/radar_tracking_results/videos/temp"
        video_name = f'{photos_file_path[:-4]}{self.filename}.avi'
        images_to_video_opencv(photos_file_path, video_name, self.fps)
        print(f"\nSaving the video to {video_name}")
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
                    measurement_dict[measurement.timestamp] = [[measurement.value[0], measurement.value[1], measurement_color]]
                else:
                    measurement_dict[measurement.timestamp].append([measurement.value[0], measurement.value[1], measurement_color])

        # Reads out the tracks and save them to a directory
        color_idx = 0
        
        #colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0'] # Orange, blå, grønn, rød, rosa
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#DBFF33','#33FFBD','#FFBD33',] # Orange, blå, grønn, rød, rosa,blå, gul/grønn, turkis, gul
        color = None
        track_dict = {}
        track_dict["Info"] = ["Track index","x","y","covariance","color"]
        for index, trajectory in track_history.items():
            # print(index)
            # print(trajectory)
            # print("\n")
            # Assigning a color to the track
            if color is not None:
                selected_color = color
            else:
                selected_color = colors[color_idx%len(colors)] 
                color_idx += 1
            # Adding the track to the dictionary
            #print("\n")
            for track in trajectory:
                #print(track)
                if track.timestamp not in track_dict:
                    track_dict[track.timestamp] = [[index, track.posterior[0][0], track.posterior[0][2], track.posterior[1][0:3:2,0:3:2], selected_color]]
                else:
                    track_dict[track.timestamp].append([index, track.posterior[0][0],track.posterior[0][2], track.posterior[1][0:3:2,0:3:2], selected_color])

        # Saving the dictionaries
        #np.save("/home/aflaptop/Documents/radar_tracker/data/track_dict.npy",track_dict)
        #np.save("/home/aflaptop/Documents/radar_tracker/data/measurement_dict.npy",measurement_dict)
        #save_track_to_mat(track_dict, "/home/aflaptop/Documents/radar_tracker/data/track_dict.mat")
        #save_measurement_to_mat(measurement_dict, "/home/aflaptop/Documents/radar_tracker/data/measurement_dict.mat")
        return measurement_dict, track_dict


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

