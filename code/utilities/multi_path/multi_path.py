import numpy as np
import matplotlib.pyplot as plt
import os
import re

class MultiPathParent:
    def __init__(self):
        self.timestamp = 0
        self.x = 0
        self.y = 0
        self.r = 0
        self.theta_min = 0
        self.theta_max = 0
        self.theta = 0
        self.cluster_area = 0
        self.error_margin_degrees = 6
        self.error_margin_radians = np.deg2rad(self.error_margin_degrees)
    
    def add_measurement(self, timestamp, measurement):
        self.timestamp = timestamp
        self.x = measurement[0]
        self.y = measurement[1]
        self.cluster_area = measurement[2]

        # polar coordinates
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)
        
        self.theta_min = self.theta - self.error_margin_radians
        self.theta_max = self.theta + self.error_margin_radians

    def plot_parent(self,ax):
        ax.plot(self.x, self.y, marker="o", color="black")
        ax.plot([0, 2*self.r*np.cos(self.theta_min)], [0, 2*self.r*np.sin(self.theta_min)], color="black", linestyle="--", linewidth=0.5,alpha=0.5)
        ax.plot([0, 2*self.r*np.cos(self.theta_max)], [0, 2*self.r*np.sin(self.theta_max)], color="black", linestyle="--", linewidth=0.5,alpha=0.5)

class MultiPathChild:
    def __init__(self, measurement):
        self.x = measurement[0]
        self.y = measurement[1]
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)

    def plot_child(self, ax):
        ax.plot(self.x, self.y, marker="o", color="red")

class MultiPath:
    def __init__(self,wokring_directory):
        self.wokring_directory = wokring_directory
        self.multi_path_scenarios = {}
        

    def add_multi_path(self, multi_path_parent, multi_path_child):
        if not multi_path_parent.timestamp in self.multi_path_scenarios:
            self.multi_path_scenarios[multi_path_parent.timestamp] = []
            self.multi_path_scenarios[multi_path_parent.timestamp].append(multi_path_parent)

        self.multi_path_scenarios[multi_path_parent.timestamp].append(multi_path_child)

    def plot_multi_path_together_with_track(self, filename, plot_next_to_multi_path, measurements_history, track_history, timestamps_history):
        fig, (ax,ax2) = plt.subplots(1,2,figsize=(20, 7.166666))
        ax.set_xlim(-120, 120)
        ax.set_ylim(-140, 20)
        ax.grid(True)
        ax2.set_xlim(-120, 120)
        ax2.set_ylim(-140, 20)
        ax2.grid(True)

        for timestamp, multi_path_scenario in self.multi_path_scenarios.items():
            for multi_path in multi_path_scenario:
                if isinstance(multi_path, MultiPathParent):
                    multi_path.plot_parent(ax)
                else:
                    multi_path.plot_child(ax)

        plot_next_to_multi_path.create_to_multi_path(ax2, measurements_history, track_history, timestamps_history)
        # save_path = "/home/aflaptop/Documents/radar_tracker/code/utilities/multi_path/multi_path_plots/for_visual_inspection"
        save_path = f"{self.wokring_directory}/code/utilities/multi_path/multi_path_plots/for_visual_inspection"
        file_name = filename.split(".")[0] + ".png"
        file_name = file_name.split("/")[-1]
        print(file_name)
        plt.savefig(os.path.join(save_path, file_name))
        plt.close

    def plot_multi_path(self, filename):
        fig, ax = plt.subplots(figsize=(10, 7.166666))
        ax.set_xlim(-120, 120)
        ax.set_ylim(-140, 20)
        ax.grid(True)

        for timestamp, multi_path_scenario in self.multi_path_scenarios.items():
            for multi_path in multi_path_scenario:
                if isinstance(multi_path, MultiPathParent):
                    multi_path.plot_parent(ax)
                else:
                    multi_path.plot_child(ax)

        # save_path = "/home/aflaptop/Documents/radar_tracker/code/utilities/multi_path/multi_path_plots"
        save_path = f"{self.wokring_directory}/code/utilities/multi_path/multi_path_plots"
        file_name = filename.split(".")[0] + ".png"
        file_name = file_name.split("/")[-1]
        print(file_name)
        plt.savefig(os.path.join(save_path, file_name))
        plt.close
    
    def valid_multi_path(self):
        if len(self.multi_path_scenarios.keys()) > 3:
            return True


def check_for_multi_path(wokring_directory, filename, plot_next_to_multi_path, measurements_history, track_history, timestamps_history, plot_statement=False):
    print("Checking for multi path")
    # measurements_dict = np.load("/home/aflaptop/Documents/radar_tracker/code/npy_files/measurement_dict.npy",allow_pickle=True).item()
    measurements_dict = np.load(f"{wokring_directory}/code/npy_files/measurement_dict.npy",allow_pickle=True).item()
 
    potential_multi_paths = []
    for timestamp, measurements in measurements_dict.items():
        if timestamp == "Info":
            continue
    
        multi_path_parent = MultiPathParent()
        for measurement in measurements:
            lenght_from_origin = np.sqrt(measurement[0]**2 + measurement[1]**2)
            cluster_area = measurement[2]
            if lenght_from_origin < 50:
                
                if cluster_area > 150:
                    multi_path_parent.add_measurement(timestamp, measurement)
                    potential_multi_paths.append(multi_path_parent)
                    
    multi_path = MultiPath(wokring_directory)
    for multi_path_parent in potential_multi_paths:
        timestamp = multi_path_parent.timestamp

        measurment_inside_sector = []
        for measurements in measurements_dict[timestamp]:
            multi_path_child = MultiPathChild(measurements)


            if multi_path_parent.theta_min < multi_path_child.theta < multi_path_parent.theta_max:
                if multi_path_child.r > multi_path_parent.r:
                    measurment_inside_sector.append(multi_path_child)
                    multi_path.add_multi_path(multi_path_parent, multi_path_child)


    if multi_path.valid_multi_path():
        if plot_statement:
            multi_path.plot_multi_path(filename)
            multi_path.plot_multi_path_together_with_track(filename, plot_next_to_multi_path, measurements_history, track_history, timestamps_history)

        return True
    else:
        return False      

# if __name__ == "__main__":
#     check_for_multi_path()