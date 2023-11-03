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

# define font size, and size of plots
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.figsize'] = 7.16666, 7.166666

class RectangleA:
    def __init__(self, bottom_left=[-80,-10], top_right=[-30,40]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 0

    def __repr__(self):
        return f"RectangleA"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleB:
    def __init__(self, bottom_left=[10,0], top_right=[50,30]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 1

    def __repr__(self):
        return f"RectangleB"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleC:
    def __init__(self, bottom_left=[40,60], top_right=[100,120]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 2

    def __repr__(self):
        return f"RectangleC"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleD:
    def __init__(self, bottom_left=[-10,80], top_right=[40,120]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 3

    def __repr__(self):
        return f"RectangleD"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleE:
    def __init__(self, bottom_left=[30,40], top_right=[60,60]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 4

    def __repr__(self):
        return f"RectangleE"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleF:
    def __init__(self, bottom_left=[-30,-10], top_right=[0,10]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 5

    def __repr__(self):
            return f"RectangleF"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class ScenarioPlot(object):
    """
    A class representing a plot depicitng the tracking scenario.
    """
    def __init__(self, measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=False, add_validation_gates=False, add_track_indexes=False, gamma=3.5,filename="coord_69",):
        self.track_marker_size = track_marker_size
        self.measurement_marker_size = measurement_marker_size
        self.add_track_indexes = add_track_indexes
        self.add_validation_gates = add_validation_gates
        self.add_covariance_ellipses = add_covariance_ellipses
        self.gamma = gamma
        self.filename = filename.split("/")[-1].split("_")[-1].split(".")[0]
        self.fig, self.ax = plt.subplots()
        self.count_matrix = np.load("/home/aflaptop/Documents/radar_tracker/data/count_matrix.npy")

    def create(self, measurements, track_history, ownship, timestamps, ground_truth=None):
        for key in ownship.keys():
            x_radar = ownship[key][0].posterior[0][0]
            y_radar = ownship[key][0].posterior[0][2]
            self.ax.scatter(x_radar,y_radar,c="black",zorder=10)
            self.ax.annotate(f"Radar",(x_radar + 2,y_radar + 2),zorder=10)
        plot_measurements(self.filename,measurements, self.ax, timestamps, marker_size=self.measurement_marker_size)
        if ground_truth:
            plot_track_pos(ground_truth, self.ax, color='k', marker_size=self.track_marker_size)

        plot_track_pos(
            track_history,
            self.ax,
            add_index=self.add_track_indexes,
            add_covariance_ellipses=self.add_covariance_ellipses,
            add_validation_gates=self.add_validation_gates,
            gamma=self.gamma)

        N_min, N_max, E_min, E_max = find_track_limits(track_history)
        self.ax.set_xlim(E_min, E_max)
        self.ax.set_ylim(N_min, N_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East [m]')
        self.ax.set_ylabel('North [m]')

        for key in track_history.keys():
            x_start = track_history[key][0].posterior[0][0]
            y_start = track_history[key][0].posterior[0][2]
            self.ax.scatter(x_start,y_start,c="red",zorder=10)
            self.ax.annotate(f"Start Track {key}",(x_start,y_start),zorder=10)

        self.ax.grid(True)
        self.fig.savefig(f'/home/aflaptop/Documents/radar_tracker/results/tracker_{self.filename}.png',dpi=600)
        print(f"Saving tracker_{self.filename}.png")
        plt.close()

    def create_video(self, measurements, track_history, ownship, timestamps, ground_truth=None):

        bar = progressbar.ProgressBar(maxval=len(measurements)).start()

        # Creating a dot where the radar is placed
        for i, key in enumerate(ownship.keys()):
            x_radar = ownship[key][0].posterior[0][0]
            y_radar = ownship[key][0].posterior[0][2]
            self.ax.scatter(x_radar,y_radar,c="black",zorder=10)
            self.ax.annotate(f"Radar",(x_radar + 2,y_radar + 2),zorder=10)

        N_min, N_max, E_min, E_max = find_track_limits(track_history)
        self.ax.set_xlim(E_min, E_max)
        self.ax.set_ylim(N_min, N_max)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East [m]')
        self.ax.set_ylabel('North [m]')
        

        interval = (timestamps-timestamps[0]+timestamps[-1]/5)/(timestamps[-1]-timestamps[0]+timestamps[-1]/5)
        positions = np.zeros((2,len(measurements)))
        color2 = None
        len_of_tracks = {}
        for i,(measurement,timestamp) in enumerate(zip(measurements,timestamps)):
            if measurement:
                cmap = get_cmap('Greys')
                x = list(measurement)[0].mean[0]
                y = list(measurement)[0].mean[1]
                positions[0][i] = x
                positions[1][i] = y
                color = cmap((interval[i].squeeze()))
                self.ax.plot(x,y, marker='o', color=color, markersize=self.measurement_marker_size)
                
                measurement_timestamp = list(measurement)[0].timestamp

                if i > 1:
                    color_idx = 0
                    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
                    
                    # The locking in this part is a bit confusing, 
                    # but it is done this way to not plot thing twice, meaning the colors get "thick"
                    for index, trajectory in track_history.items():
                        if len(trajectory) == 0:
                            continue

                        if color2 is not None:
                            selected_color = color2
                        else:
                            selected_color = colors[color_idx%3]
                            color_idx += 1

                        # Used to check if there has been added any new tracks
                        plot_away = True
                        try:
                            temp1 = len_of_tracks[index]
                        except KeyError:
                            temp1 = None

                        # adding track position and ellipse info to list
                        track_pos = []
                        ellipse_center = []
                        ellipse_sigma = []
                        for track in trajectory:
                            if track.timestamp < measurement_timestamp:
                                track_pos.append(track.posterior[0])
                                ellipse_center.append(track.posterior[0][0:3:2]),
                                ellipse_sigma.append(track.posterior[1][0:3:2,0:3:2])
                            else:
                                break
                        
                        # Adding the lenght of the different tracks to a dictionary
                        len_of_tracks[index] = len(track_pos) 
                        try:
                            temp2 = len_of_tracks[index]
                        except KeyError:
                            temp2 = None
                        # Checking if the length from last iteration is equal
                        # If it is equal, that means we have no new data to add to this track, 
                        # and it should thefore not be plotted.
                        if temp1 == temp2:
                            plot_away = False
                        
                        if plot_away:
                            # Need to check if we have any elemnts to plot
                            if track_pos:
                                track_pos = np.array(track_pos)
                                # plotting only the last two elemnts, since the others have been plottet previos
                                line, = self.ax.plot(track_pos[len(track_pos)-2:,0], track_pos[len(track_pos)-2:,2], color=selected_color, lw=1,ls="-")

                            if self.add_covariance_ellipses:
                                if ellipse_center:
                                    edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0)
                                    facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
                                    # Plotting the last element in the ellipse_center list since the others have been plottet earlier
                                    covariance_ellipse = get_ellipse(ellipse_center[-1], ellipse_sigma[-1])
                                    self.ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))

                self.ax.set_title(f"Time: {timestamp[0]}")
                self.fig.savefig(f'/home/aflaptop/Documents/radar_tracker/results/videos/temp/tracker_{i+1}.png',dpi=100)
                bar.update(i)

        video_name = f'/home/aflaptop/Documents/radar_tracker/results/videos/{self.filename}.avi'
        images_to_video_opencv('/home/aflaptop/Documents/radar_tracker/results/videos/temp', video_name, 1)
        print(f"Saving {video_name.split('/')[-1]}")
        empty_folder("/home/aflaptop/Documents/radar_tracker/results/videos/temp")

    def update_count(rectangles):
        data = {"RectangleA": 0, "RectangleB": 0, "RectangleC": 0, "RectangleD": 0, "RectangleE": 0, "RectangleF": 0}
        for rectangle in rectangles:
            data[rectangle] += 1
        with open('/home/aflaptop/Documents/radar_tracker/data/count.yaml', 'w') as file:
            yaml.dump(data, file)

    def check_start_and_stop_2(self, track_history):
        
        print(self.count_matrix)
        rectangleA = RectangleA()
        rectangleB = RectangleB()
        rectangleC = RectangleC()
        rectangleD = RectangleD()
        rectangleE = RectangleE()
        rectangleF = RectangleF()
        rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF]  
        for index, trajectory in track_history.items():
            x_start = track_history[index][0].posterior[0][0]
            y_start = track_history[index][0].posterior[0][2]
            x_stop = track_history[index][-1].posterior[0][0]
            y_stop = track_history[index][-1].posterior[0][2]
            rectangles = []
            for rectangle in rectangles:
                # Start
                if rectangle.start_or_stop(x_start,y_start):
                    rectangles.append(rectangle)
                    print(f"Track {index} started in rectangle {rectangle}")

                # Stop
                if rectangle.start_or_stop(x_stop,y_stop):
                    rectangles.append(rectangle)
                    print(f"Track {index} stopped in rectangle {rectangle}")

    def check_start_and_stop(self, track_history):
        rectangleA = RectangleA()
        rectangleB = RectangleB()
        rectangleC = RectangleC()
        rectangleD = RectangleD()
        rectangleE = RectangleE()
        rectangleF = RectangleF()
        rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF]  
        start_rectangle = {}
        stop_rectangle = {}
        for index, trajectory in track_history.items():
            x_start = track_history[index][0].posterior[0][0]
            y_start = track_history[index][0].posterior[0][2]
            x_stop = track_history[index][-1].posterior[0][0]
            y_stop = track_history[index][-1].posterior[0][2]
            for rectangle in rectangles:
                # Start
                if rectangle.start_or_stop(x_start,y_start):
                    start_rectangle[index] = rectangle
                # Stop
                if rectangle.start_or_stop(x_stop,y_stop):
                    stop_rectangle[index] = rectangle

        for start_key in start_rectangle.keys():
            if start_key in stop_rectangle.keys():
                self.count_matrix[stop_rectangle[start_key].index][start_rectangle[start_key].index] += 1
        np.save("/home/aflaptop/Documents/radar_tracker/data/count_matrix.npy",self.count_matrix)
        # print(start_rectangle)
        # print(stop_rectangle)
        # print(self.count_matrix)

    def reset_count_matrix(self):
        self.count_matrix = np.zeros((6,6))
        np.save("/home/aflaptop/Documents/radar_tracker/data/count_matrix.npy",self.count_matrix)

def plot_measurements(filename,measurements_all, ax, timestamps, marker_size=5):
    cmap = get_cmap('Greys')
    measurements_all = dict((i, set(measurements)) for i, measurements in enumerate(measurements_all))

    timestamps = np.asarray(timestamps)
    interval = (timestamps-timestamps[0]+timestamps[-1]/5)/(timestamps[-1]-timestamps[0]+timestamps[-1]/5)
    for index, measurement_set in measurements_all.items():
        try:
            color = cmap(interval[index].squeeze())
        except Exception as e:
            print(f"Error {e} with file: {filename}")
        for measurement in measurement_set:
            ax.plot(measurement.value[0], measurement.value[1], marker='o', color=color, markersize=marker_size)

def plot_track_pos(track_history, ax, add_index=False, add_covariance_ellipses=False, add_validation_gates=False, gamma=3.5, lw=1, ls='-', marker_size = 5, color=None,):
    color_idx = 0
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0'] # Orange, blå, grønn, rød, rosa
    #colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    for index, trajectory in track_history.items():
        if len(trajectory) == 0:
            continue

        positions = np.array([track.posterior[0] for track in trajectory])

        if color is not None:
            selected_color = color
        else:
            selected_color = colors[color_idx%5] # colors[color_idx%3]
            color_idx += 1

        line, = ax.plot(positions[:,0], positions[:,2], color=selected_color, lw=lw,ls=ls)
        last_position, = ax.plot(positions[-1,0], positions[-1,2], 'o', color=selected_color, markersize=marker_size)

        if add_covariance_ellipses:
            edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0)
            facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
            #print(trajectory)
            for track in trajectory:
                covariance_ellipse = get_ellipse(track.posterior[0][0:3:2], track.posterior[1][0:3:2,0:3:2])
                ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))

        if add_validation_gates:
            edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.5)
            facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
            for track in trajectory[1:]: # the first track estimate has no validation gate
                validation_gate = get_validation_gate(track, gamma=gamma)
                ax.add_patch(PolygonPatch(validation_gate, facecolor = facecolor, edgecolor = edgecolor))

        if add_index:
            ax.text(positions[-1,0], positions[-1,2]-5, str(index), color='black')

def get_validation_gate(state, gamma):
        for j, predicted_measurement in enumerate(state.predicted_measurements.leaves):
            validation_gate = get_ellipse(predicted_measurement.mean, predicted_measurement.covariance, gamma)
            if j == 0:
                union = validation_gate
            else:
                if validation_gate.intersection(union).is_empty:
                    warnings.warn(f'The validation gates for the kinematic pdfs for track {state.index} at time {state.timestamp} are disjoint. The displayed validation gate will be displayed incorrectly.')
                    pass
                else:
                    union = validation_gate.union(union)
        return union

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

def old_shit():
    print("hehe")
    # for index, trajectory in track_history.items():
                    #     if self.add_covariance_ellipses:
                    #         ellipse_center = []
                    #         ellipse_sigma = []
                    #         for track in trajectory:
                    #             if track.timestamp < measurement_timestamp:
                    #                 ellipse_center.append(track.posterior[0][0:3:2])
                    #                 ellipse_sigma.append(track.posterior[1][0:3:2,0:3:2])
                    #             else:
                    #                 break
                    #         if ellipse_center:
                    #             edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0)
                    #             facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
                    #             for i in range(len(ellipse_center)):
                    #                 covariance_ellipse = get_ellipse(ellipse_center[i], ellipse_sigma[i])
                    #                 self.ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))

                            # try:
                            #     #print(trajectory[i].posterior[0][0:3:2])
                            #     print((trajectory[i]))
                            #     covariance_ellipse = get_ellipse(trajectory[i].posterior[0], trajectory[i].posterior[1][0:3:2,0:3:2])
                            #     #covariance_ellipse = get_ellipse(track_pos[0:3:2], trajectory[i].posterior[1][0:3:2,0:3:2])
                            #     self.ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))
                            # except Exception as e:
                            #     print(e)
                        # for track in trajectory:
                        #     print(track)
                        #     print("########################")
                        # print("-----------------------------------------")

                        # positions2 = np.array([track.posterior[0] for track in trajectory])
                        # line, = self.ax.plot(positions2[:i,0], positions2[:i,2], color=selected_color, lw=1,ls="-")
                        # if i == len(measurements) - 1:
                        #     last_position, = self.ax.plot(positions2[-1,0], positions2[-1,2], 'o', color=selected_color, markersize=self.measurement_marker_size)

                        # if self.add_covariance_ellipses:
                        #     edgecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0)
                        #     facecolor = matplotlib.colors.colorConverter.to_rgba(selected_color, alpha=0.16)
                        #     #for track in list(trajectory[i]):
                        #     #print(trajectory[i])
                        #     try:
                        #         covariance_ellipse = get_ellipse(center=trajectory[i].posterior[0][0:3:2], Sigma=trajectory[i].posterior[1][0:3:2,0:3:2])
                        #         self.ax.add_patch(PolygonPatch(covariance_ellipse, facecolor = facecolor, edgecolor = edgecolor))
                        #     except Exception as e:
                        #         print(f"Exception: {e}")