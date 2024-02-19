import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

from matplotlib.cm import get_cmap
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry import Polygon
from descartes import PolygonPatch
import datetime
from parameters import tracker_params, measurement_params, process_params, tracker_state
from matplotlib.colors import LinearSegmentedColormap

# define font size, and size of plots
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.figsize'] = 7.16666, 7.166666

class ScenarioPlot(object):
    """
    A class representing a plot depicitng the tracking scenario.
    """
    def __init__(self, measurement_marker_size=3, track_marker_size=5, add_covariance_ellipses=False, add_validation_gates=False, add_track_indexes=False, gamma=3.5,filename="coord_69",dir_name="test",resolution=600):
        self.track_marker_size = track_marker_size
        self.measurement_marker_size = measurement_marker_size
        self.add_track_indexes = add_track_indexes
        self.add_validation_gates = add_validation_gates
        self.add_covariance_ellipses = add_covariance_ellipses
        self.gamma = gamma
        self.resolution = resolution
        self.filename = filename.split("/")[-1].split("_")[-1].split(".")[0]
        self.dir_name = dir_name
        self.fig, self.ax = plt.subplots(figsize=(11, 7.166666))
        self.ax1 = self.ax
        self.write_parameters_to_plot()
        

    def create(self, measurements, track_history, ownship, timestamps, ground_truth=None):

        self.write_track_time_to_plot(track_history)
        self.write_coherence_factor_to_plot(track_history)

        # Radar pos
        x_radar = 0
        y_radar = 0
        self.ax1.scatter(x_radar,y_radar,c="black",zorder=10)
        self.ax1.annotate(f"Radar",(x_radar + 2,y_radar + 2),zorder=10)
        
        plot_measurements(self.filename,measurements, self.ax1, timestamps, marker_size=self.measurement_marker_size)
        
        if ground_truth:
            plot_track_pos(ground_truth, self.ax1, color='k', marker_size=self.track_marker_size)
        
        plot_track_pos(
            track_history,
            self.ax1,
            add_index=self.add_track_indexes,
            add_covariance_ellipses=self.add_covariance_ellipses,
            add_validation_gates=self.add_validation_gates,
            gamma=self.gamma)

        #N_min, N_max, E_min, E_max = find_track_limits(track_history)
        self.ax1.set_xlim(-120, 120)
        self.ax1.set_ylim(-140, 20)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlabel('East [m]')
        self.ax1.set_ylabel('North [m]')
        plt.tight_layout()

        for key in track_history.keys():
            x_start = track_history[key][0].posterior[0][0]
            y_start = track_history[key][0].posterior[0][2]
            self.ax1.scatter(x_start,y_start,c="red",zorder=10)
            self.ax1.annotate(f"Start Track {key}",(x_start,y_start),zorder=10)

        self.ax1.grid(True)
        #plt.show()
        now_time = datetime.datetime.now().strftime("%H,%M,%S")
        save_name = f'{self.dir_name}/{self.filename}({now_time}).png'
        self.fig.savefig(save_name,dpi=self.resolution)
        print(f"Saving tracker_{save_name}")
        plt.close()

    def create_with_map(self, measurements, track_history, ownship, timestamps, ground_truth=None):

        # Plotting the occupancy grid'
        data = np.load(f"/home/aflaptop/Documents/radar_tracker/code/npy_files/occupancy_grid.npy",allow_pickle='TRUE').item()
        occupancy_grid = data["occupancy_grid"]
        origin_x = data["origin_x"]
        origin_y = data["origin_y"]

        colors = [(1, 1, 1), (0.8, 0.8, 0.8)]  # Black to light gray
        cm = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
        self.ax1.imshow(occupancy_grid, cmap=cm, interpolation='none', origin='upper', extent=[0, occupancy_grid.shape[1], 0, occupancy_grid.shape[0]])
        

        self.write_track_time_to_plot(track_history)
        self.write_coherence_factor_to_plot(track_history)

        # Radar pos
        x_radar = origin_x
        y_radar = origin_y
        self.ax1.plot(x_radar,y_radar,c="red", marker="o", zorder=10, markersize=10)
        self.ax1.annotate(f"Radar",(x_radar + 2,y_radar + 2),zorder=10,fontsize=10)

        plot_measurements(self.filename,measurements, self.ax1, timestamps, marker_size=self.measurement_marker_size)
        
        if ground_truth:
            plot_track_pos(ground_truth, self.ax1, color='k', marker_size=self.track_marker_size)

        plot_track_pos(
            track_history,
            self.ax1,
            add_index=self.add_track_indexes,
            add_covariance_ellipses=self.add_covariance_ellipses,
            add_validation_gates=self.add_validation_gates,
            gamma=self.gamma)

        #N_min, N_max, E_min, E_max = find_track_limits(track_history)
        self.ax1.set_xlim(origin_x-120,origin_x + 120)
        self.ax1.set_ylim(origin_y-140, origin_y + 20)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlabel('East [m]')
        self.ax1.set_ylabel('North [m]')
        plt.tight_layout()

        for key in track_history.keys():
            x_start = track_history[key][0].posterior[0][0]
            y_start = track_history[key][0].posterior[0][2]
            self.ax1.plot(x_start,y_start,c="red", marker="o",zorder=10,markersize=5)
            self.ax1.annotate(f"Start Track {key}",(x_start,y_start),zorder=10)

        # reformating the x and y axis
        x_axis_list = np.arange(origin_x-120,origin_x+121,20)
        x_axis_list_str = []
        for x in x_axis_list:
            x_axis_list_str.append(str(int(x-origin_x)))
        plt.xticks(x_axis_list, x_axis_list_str)

        y_axis_list = np.arange(origin_y-140,origin_y+21,20)
        y_axis_list_str = []
        for y in y_axis_list:
            y_axis_list_str.append(str(int(y-origin_y)))
        plt.yticks(y_axis_list, y_axis_list_str)

        self.ax1.grid(True)
        # plt.show()
        now_time = datetime.datetime.now().strftime("%H,%M,%S")
        save_name = f'{self.dir_name}/{self.filename}({now_time}).png'
        self.fig.savefig(save_name,dpi=self.resolution)
        print(f"Saving tracker_{save_name}")
        plt.close()


    def write_track_time_to_plot(self, track_history):
        text = ""
        for i,track in enumerate(track_history.items()):
            track_id = track[0]
            track_start_time = track[1][0].timestamp
            track_end_time = track[1][-1].timestamp
            track_time = track_end_time - track_start_time
            text += f"Track {track_id} time: {track_time:.2f}"
            if i < len(track_history.items()) - 1:
                text += "\n"

        self.ax.text(0.0, -0.13, text, transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='grey', alpha=0.15))

    def write_coherence_factor_to_plot(self, track_history,coherence_factor=0.75):
        text = "Coherence factor for tracks:\n"
        for i,track in enumerate(track_history.items()):
            u = []
            v = []
            last_mean, cov = track[1][0].posterior
            for k,track_point in enumerate(track[1]):
                if k == 0:
                    continue
                mean, cov = track_point.posterior
                u.append([[mean[0]-last_mean[0]],[mean[2]-last_mean[2]]])
                v.append([mean[1],mean[3]])
                last_mean = mean
            u = np.array(u)
            v = np.array(v)

            ck = 0
            for k in range(1,len(u)):
                c_k = np.dot(np.transpose(v[k]),u[k])/(np.linalg.norm(u[k])*np.linalg.norm(v[k]))
                ck += c_k[0]
            text += f"Track {track[0]} = {ck/len(u):.2f}"

            if i < len(track_history.items()) - 1:
                text += "\n"

        self.ax.text(0.3, -0.13, text, transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='grey', alpha=0.15))
            #print(f"Coherence factor for track {track[0]} = {ck/len(u):.2f}\n")
            #if ck/len(u) < coherence_factor:
            #    not_valid_tracks.append(track[0])

        #print(f"Tracks with to low coherence factor: {not_valid_tracks}\n")
        #return not_valid_tracks

    def write_parameters_to_plot(self):
        my_text0= f"IMM_off: {tracker_state['IMM_off']} \n"
        my_text1= f"Tracker parameters: \n"
        my_text2 = f"Maximum velocity: {tracker_params['maximum_velocity']} \n"
        my_text3 = f"Init Pvel: {np.sqrt(tracker_params['init_Pvel'])} \n"
        my_text4 = f"P_D: {tracker_params['P_D']} \n"
        my_text5 = f"Clutter density: {tracker_params['clutter_density']} \n"
        my_text6 = f"Gamma: {np.sqrt(tracker_params['gamma'])} \n"
        my_text7 = f"Survival prob: {tracker_params['survival_prob']} \n"
        my_text8 = f"Birth intensity: {tracker_params['birth_intensity']} \n"
        my_text9 = f"Init prob: {tracker_params['init_prob']:.4f} \n"
        my_text10 = f"Conf threshold: {tracker_params['conf_threshold']} \n"
        my_text11 = f"Term threshold: {tracker_params['term_threshold']} \n"
        my_text12 = f"Cart cov: {np.sqrt(measurement_params['cart_cov'][0][0])} \n"
        my_text13 = f"Range cov: {np.sqrt(measurement_params['range_cov'])} \n"
        my_text14 = f"Bearing cov: {np.sqrt(measurement_params['bearing_cov']):.5f} \n"
        my_text15 = f"Cov CV single: {np.sqrt(process_params['cov_CV_single'])}"
        #my_text16 = f"Cov CV high: {np.sqrt(process_params['cov_CV_high'])} \n"
        #my_text17 = f"Cov CT: {np.sqrt(process_params['cov_CT'])} \n"
        my_text = my_text0 + my_text1 + my_text2 + my_text3 + my_text4 + my_text5 + my_text6 + my_text7 + my_text8 + my_text9 + my_text10 + my_text11 + my_text12 + my_text13 + my_text14 + my_text15 #+ my_text16 + my_text17
        
        props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        font_size = 10
        self.ax.text(1.03, 0.99, my_text, transform=self.ax.transAxes, fontsize=font_size, verticalalignment='top', bbox=props)



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
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#DBFF33','#33FFBD','#FFBD33',] # Orange, blå, grønn, rød, rosa,blå, gul/grønn, turkis, gul
    #colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    for index, trajectory in track_history.items():
        if len(trajectory) == 0:
            continue

        positions = np.array([track.posterior[0] for track in trajectory])

        if color is not None:
            selected_color = color
        else:
            selected_color = colors[color_idx%len(colors)] # colors[color_idx%3]
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
