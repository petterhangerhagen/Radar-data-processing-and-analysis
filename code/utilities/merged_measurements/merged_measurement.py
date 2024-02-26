import scipy.io as io
import numpy as np
from tracking import constructs
from parameters import measurement_params
import json
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import os

class CloseMeasurementsPair:
    """
    Class for storing two close measurements and the distance between them
    """
    def __init__(self, timestamp, measurement1, measurement2):
        self.timestamp = timestamp
        self.measurement1 = measurement1[0:3]
        self.measurement2 = measurement2[0:3]
        self.areas = [measurement1[2], measurement2[2]]
        self.distance_between = np.sqrt((measurement1[0]-measurement2[0])**2 + (measurement1[1]-measurement2[1])**2)

    def __repr__(self) -> str:
        temp_str = f"Timestamp: {self.timestamp:.2f}\n"
        temp_str += f"Measurement 1: {self.measurement1}\n"
        temp_str += f"Measurement 2: {self.measurement2}\n"
        temp_str += f"Distance between measurements: {self.distance_between:.2f}"
        return temp_str

class MergeMeasurements:
    """
    Class for storing the previous and current measurements that are merged
    """
    def __init__(self, close_measurement_pair, current_timestamp, current_measurement):
        self.prev_timestamp = close_measurement_pair.timestamp
        self.prev_measurement1 = close_measurement_pair.measurement1
        self.prev_measurement2 = close_measurement_pair.measurement2
        self.area = close_measurement_pair.areas

        self.current_timestamp = current_timestamp
        self.current_measurement = current_measurement
        self.current_area = current_measurement[2]

    def __repr__(self) -> str:
        temp_str = f"Previous timestamp: {self.prev_timestamp:.2f}\n"
        temp_str += f"Previous measurements: {self.prev_measurements}\n"
        temp_str += f"Current timestamp: {self.current_timestamp:.2f}\n"
        temp_str += f"Current measurement: {self.current_measurement}\n"
        return temp_str
    
    def add_to_plot(self, ax):
        ax.scatter(self.prev_measurement1[0], self.prev_measurement1[1], color='red', s=self.area[0])
        ax.scatter(self.prev_measurement2[0], self.prev_measurement2[1], color='red', s=self.area[1])
        ax.scatter(self.current_measurement[0], self.current_measurement[1], color='blue', s=self.current_area)

def radar_data_json_file(json_file):
    """
    Reads out the radar data from the json file and returns the measurements, radar and timestamps.
    Used for creating the dictionaries for the measurements and tracks. The track dictionary are not used by this script.
    These measurement dictionaries have been used for a more visual representation of the measurements.
    """

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
            area = measurement["area"]
            meas_set = (np.array([x,y,area]))
            measurements[-1].add(constructs.Measurement(meas_set, measurement_params['cart_cov'],  float(timestamp)))
        timestamps.append(float(timestamp))

    measurements = np.array(measurements)
    timestamps = np.array(timestamps)

    ownship = np.zeros((len(timestamps),5))
    radar = {1: [constructs.State(ownship_pos, np.identity(4), timestamp) for ownship_pos, timestamp in zip(ownship, timestamps)]}

    timestamps = np.reshape(timestamps,(len(timestamps),1))
    return measurements, radar, timestamps

def create_dict(filename, track_history):
        """
        Creates a dictionary for the measurements and tracks and saves them to a directory.
        Where the keys are the timestamps and the values are the measurements and tracks at that timestamp.
        The measurements are saved as a list of lists with the format [x,y,area,color].
        The tracks are saved as a list of lists with the format [track index, x, y, covariance, color].
        """

        measurements, _ , timestamps = radar_data_json_file(filename)

        # Reads out the measurements and save them to a directory
        cmap = get_cmap('Greys')
        measurements_all = measurements
        measurements_all = dict((i, set(measurements)) for i, measurements in enumerate(measurements_all))
        timestamps = np.asarray(timestamps)
        interval = (timestamps-timestamps[0]+timestamps[-1]/5)/(timestamps[-1]-timestamps[0]+timestamps[-1]/5)
        measurement_dict = {}
        measurement_dict["Info"] = ["x","y","area","color"]
        for index, measurement_set in measurements_all.items():
            measurement_color = cmap(interval[index].squeeze())
            for measurement in measurement_set:
                if measurement.timestamp not in measurement_dict:
                    measurement_dict[measurement.timestamp] = []
                measurement_dict[measurement.timestamp].append([measurement.value[0], measurement.value[1],measurement.value[2], measurement_color])

        # Reads out the tracks and save them to a directory
        color_idx = 0
        #colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0'] # Orange, blå, grønn, rød, rosa
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#DBFF33','#33FFBD','#FFBD33',] # Orange, blå, grønn, rød, rosa,blå, gul/grønn, turkis, gul
        color = None
        track_dict = {}
        track_dict["Info"] = ["Track index","x","y","covariance","color"]
        for index, trajectory in track_history.items():
            if color is not None:
                selected_color = color
            else:
                selected_color = colors[color_idx%len(colors)] 
                color_idx += 1

            for track in trajectory:
                if track.timestamp not in track_dict:
                    track_dict[track.timestamp] = []
                track_dict[track.timestamp].append([index, track.posterior[0][0],track.posterior[0][2], track.posterior[1][0:3:2,0:3:2], selected_color])

        # Saving the dictionaries
        np.save("/home/aflaptop/Documents/radar_tracker/code/npy_files/track_dict.npy",track_dict)
        np.save("/home/aflaptop/Documents/radar_tracker/code/npy_files/measurement_dict.npy",measurement_dict)
       
        return measurement_dict, track_dict


def merged_measurements(filename,track_history, plot_scenarios=False, return_true_or_false=True):
    """
    Function for finding and visualizing the merged measurements.
    The function reads out the measurement dictionary and finds all measurements which are close to each other at the same timestamp.
    The following need to be satisfied for the measurements to be approved 'before' they are merged:
    - The measurements are less than 20 meters apart
    - The areas of the measurements are larger than 50
    - The measurements are inside the polygon
    The merged measurements are then visualized.
    """

    vertices = [(100, 0), (100, -40), (0, -80), (-50,-110), (-90, -120), (-105, -110),(-50,-60),(-25,-20),(0,0)]
    data = np.load("/home/aflaptop/Documents/radar_tracker/code/npy_files/measurement_dict.npy",allow_pickle=True).item()
    timestamps = list(data.keys())
    x_list = []
    y_list = []
    color_list = []
    close_measurements_dict = {}
    for k,(timestamp, measurements) in enumerate(data.items()):
        if timestamp == "Info":
            continue
        
        # Find all close measurements and approve them if they satisfy the conditions
        for i in range(len(measurements)):
            for j in range(i+1,len(measurements)):
                distance = np.sqrt((measurements[i][0]-measurements[j][0])**2 + (measurements[i][1]-measurements[j][1])**2)
                if distance < 20:

                    close_measurements = CloseMeasurementsPair(timestamp, measurements[i], measurements[j])

                    if close_measurements.areas[0] > 50 and close_measurements.areas[1] > 50:
                        if point_inside_polygon(close_measurements.measurement1, vertices):
                            if point_inside_polygon(close_measurements.measurement2, vertices):

                                if timestamp not in close_measurements_dict:
                                    close_measurements_dict[timestamp] = []
                                close_measurements_dict[timestamp].append(close_measurements)

            measurement = measurements[i]
            x_list.append(measurement[0])
            y_list.append(measurement[1])
            color_list.append(measurement[3])

    if plot_scenarios:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x_list, y_list, c=color_list)#, s=area)
        ax.set_xlim(-110,100)
        ax.set_ylim(-140,10)
        draw_polygon(vertices,ax)
    
    print("############### Merged measurements ###############")
    if len(track_history.items()) < 3:
        print("Fewer than 3 tracks, assume no merged measurments.")
        print("##################################################\n")
        return False
    
    number_of_merged_measurements = 0
    merged_measurements_dict = {}
    for k, (timestamp, close_measurement_pair_list) in enumerate(close_measurements_dict.items()):

        # Finds the index of the timestamp, so that the next timestamp can be found. Because it is the next timestamp that the measurements can be merged at.
        timestamp_index = timestamps.index(close_measurement_pair_list[0].timestamp)
        for i in range(len(close_measurement_pair_list)):
            close_measurement_pair = close_measurement_pair_list[i]

            if timestamp_index == len(timestamps)-1:
                continue
            next_timestamp = timestamps[timestamp_index+1]
            next_measurements = data[next_timestamp]
            
            for next_measurement in next_measurements:
                distances = []
                distance1 = np.sqrt((next_measurement[0]-close_measurement_pair.measurement1[0])**2 + (next_measurement[1]-close_measurement_pair.measurement1[1])**2)
                distance2 = np.sqrt((next_measurement[0]-close_measurement_pair.measurement2[0])**2 + (next_measurement[1]-close_measurement_pair.measurement2[1])**2)
                distances.append(distance1)
                distances.append(distance2)

                areas = close_measurement_pair.areas
                
                # If the next measurements are less than 20 meters away from the close measurements,
                # and the area of the next measurement is larger than the sum of the areas of the close measurements, then the measurements are merged.
                if max(distances) < 20:
                    if next_measurement[2] > sum(areas):

                        number_of_merged_measurements += 1
                        merged_measurement = MergeMeasurements(close_measurement_pair, next_timestamp ,next_measurement)
                        if plot_scenarios:
                            merged_measurement.add_to_plot(ax)
                        merged_measurements_dict[number_of_merged_measurements] = merged_measurement

                        print("Merged measurement")
                        print(close_measurement_pair)
                        print("\n")
                    
    
    print(f"Number of merged measurements: {number_of_merged_measurements}")
    if plot_scenarios:
        if number_of_merged_measurements > 0:
            file_name = os.path.basename(filename)
            file_name = os.path.splitext(file_name)[0]
            save_path = "/home/aflaptop/Documents/radar_tracker/code/utilities/merged_measurements/merged_measurements_plots"
            save_path = os.path.join(save_path,file_name + ".png")
            print(f"Saving plot to {save_path}")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.close()
            
    #plt.close()

    print("##################################################\n")
    if return_true_or_false:
        if number_of_merged_measurements > 0:
            return True
        else:
            return False


def draw_polygon(vertices,ax):
    polygon = Polygon(vertices, closed=True, fill=None, edgecolor='b')
    ax.add_patch(polygon)

def point_inside_polygon(measurement, vertices):
    """
    Function for checking if a point is inside a polygon
    """
    x = measurement[0]
    y = measurement[1]
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# if __name__ == "__main__":
#     merged_measurements()
    