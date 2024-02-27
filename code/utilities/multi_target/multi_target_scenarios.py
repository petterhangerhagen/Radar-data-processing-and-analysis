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
        # print("Timestamp: ", timestamp)
        # for track in tracks:
        #     print(f"Track {track[0]}: x = {track[1]}, y = {track[2]}, covariance = {track[3]}")
        # print("\n")

        for i in range(len(tracks)):
            for j in range(i+1,len(tracks)):
                track1, cov1 = [tracks[i][1],tracks[i][2]], tracks[i][3]
                track2, cov2 = [tracks[j][1],tracks[j][2]], tracks[j][3]
                cov1, cov2 = cov1, cov2
                distance = np.sqrt((track1[0]-track2[0])**2 + (track1[1]-track2[1])**2)
                if distance < threshold_distance:
                    #print(f"At time {timestamp:.2f}, tracks {tracks[i][0]} and {tracks[j][0]} are close to each other")
                    #print(cov1)
                    cov1_inv = np.linalg.inv(cov1)
                    cov2_inv = np.linalg.inv(cov2)
                    dxy = np.array([track2[0] - track1[0], track2[1] - track1[1]])
                    mahalanobis1 = np.sqrt(np.dot(np.dot(dxy.T, cov1_inv), dxy))
                    mahalanobis2 = np.sqrt(np.dot(np.dot(dxy.T, cov2_inv), dxy))
                    if mahalanobis1 <= n or mahalanobis2 <= n:
                    #if np.dot(np.dot(dxy, cov1_inv), dxy) <= 1 or np.dot(np.dot(dxy, cov2_inv), dxy) <= 1:
                        # print(f"At time {timestamp:.2f} multi-target scenario detected. between tracks {tracks[i][0]} and {tracks[j][0]}")
                        
                        # #print(f"MD = {np.dot(np.dot(dxy, cov1_inv), dxy)}")
                        # print(f"MD1 = {mahalanobis1:.2f}, MD2 = {mahalanobis2:.2f}")
                        # print(f"cov1 = {cov1}, cov2 = {cov2}")
                        # print(f"distance = {distance:.2f}")
                        # print(f"track1 = {track1}, track2 = {track2}")
                        # fig, ax = plt.subplots()
                        # ax.scatter(track1[0], track1[1], c='b', label=f'Track {tracks[i][0]}',zorder=10)
                        # ax.scatter(track2[0], track2[1], c='r', label=f'Track {tracks[j][0]}',zorder=10)
                        # ellipse1 = get_ellipse(track1, cov1)
                        # ellipse2 = get_ellipse(track2, cov2)
                        # ax.add_patch(PolygonPatch(ellipse1))
                        # ax.add_patch(PolygonPatch(ellipse2))
                        # ax.legend()
                        # a = cov1[0,0]
                        # b = cov1[0,1]
                        # c = cov1[1,1]
                        # lambda_1 = (a + c)/2 + np.sqrt(((a-c)/2)**2 + b**2)
                        # lambda_2 = (a + c)/2 - np.sqrt(((a-c)/2)**2 + b**2)
                        # if b == 0 and a >= c:
                        #     angle = 0
                        # elif b == 0 and a < c:
                        #     angle = np.pi/2
                        # else:
                        #     angle = np.arctan2(lambda_1-a, b)
                        # print(f"angle = {np.rad2deg(angle):.2f}")
                        # print(f"lambda_1 = {np.sqrt(lambda_1):.2f}, lambda_2 = {np.sqrt(lambda_2):.2f}")
                        # x , y = np.sqrt(lambda_1)*np.cos(angle),  np.sqrt(lambda_1)*np.sin(angle)
                        # x,y = 1.8*x, 1.8*y
                        # print(track1 )
                        # ax.plot([track1[0], (x+track1[0])],[track1[1],(y+track1[1])], c='red',zorder=10)
                        #x = [ (track1[0] + np.sqrt(lambda_1)*np.cos(angle)), track2[0] + np.sqrt(lambda_2)*np.cos(angle)]
                        #y = [ (track1[1] + np.sqrt(lambda_1)*np.sin(angle)), track2[1] + np.sqrt(lambda_2)*np.sin(angle)]
                        #y = [track1[1], (track1[0] + cov1[1,1])]#, track2[1] + cov2[1,1]]
                        #ax.scatter(x,y, c='red',zorder=10)

                        #plt.show()
                        return True
    return False

def move_plot_to_this_directory(wokring_directory, filename, dirname):
    source_dir = dirname
    # destaination_dir = "/home/aflaptop/Documents/radar_tracker/code/utilities/multi_target/multi_target_plots"
    destaination_dir = f"{wokring_directory}/code/utilities/multi_target/multi_target_plots"
    partial_file_name = os.path.basename(filename)
    partial_file_name = partial_file_name.split(".")[0]
    partial_file_name = partial_file_name.split("_")[-1]

    files_in_dir = os.listdir(source_dir)
    # Find the matching file
    # Initialize an empty list to store matching files
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

# if __name__ == "__main__":
#     dir_name = "/home/aflaptop/Documents/radar_tracking_results/20-Feb"
#     file_name = "/home/aflaptop/Documents/radar_data/data_sep_8-9-11-14/rosbag_2023-09-09-11-05-32.json"
#     move_plot_to_this_directory(file_name, dir_name)