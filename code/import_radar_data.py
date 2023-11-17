import scipy.io as io
import numpy as np
from tracking import constructs
from parameters import measurement_params
import csv
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def NE_to_xy(state_list):
    """
    Convert from North-East coordinates to xy-coordinates.
    """
    if len(state_list[0]) == 2:
        transform = np.array([[0,1],[1,0]])
    elif len(state_list[0]) == 4:
        transform = np.array([[0,0,1,0],
                              [0,0,0,1],
                              [1,0,0,0],
                              [0,1,0,0]])
    elif len(state_list[0]) == 5:
        transform = np.array([[0,0,1,0,0],
                              [0,0,0,1,0],
                              [1,0,0,0,0],
                              [0,1,0,0,0],
                              [0,0,0,0,0]])
    else:
        print('Wrong data input shape.')
        exit()

    return np.matmul(state_list, transform)

def ensure_correct_state_dimension(state_list):
    pad_amount = len(measurement_params['measurement_mapping'][0])-len(state_list[0])
    return np.pad(state_list, [(0, pad_amount), (0, pad_amount)])

def radar_data_csv_file(filename):
    data = []
    with open(filename,"r") as csvfile:
        csvreader  = csv.reader(csvfile)
        for row in csvreader:
            data.append(row[1:])
        data.pop(0)

    timestamps = []
    measurements = []
    for i, elem in enumerate(data):
        current_timestamp = data[i][0]
        y = float(data[i][1])*(-1)
        x = float(data[i][2])*(-1)
        if current_timestamp not in timestamps:
            measurements.append([np.array([x,y])])
            timestamps.append(current_timestamp)
        else:
            ind = timestamps.index(current_timestamp)
            measurements[ind].append(np.array([x,y]))
    
    for i, elem in enumerate(timestamps):
        date_object = datetime.strptime(elem, "%Y-%m-%d %H:%M:%S")
        timestamp = datetime.timestamp(date_object)
        if i==0:
            first_timestamp = timestamp
            timestamps[i] = float(0)
        else:
            timestamps[i] = float(timestamp)-float(first_timestamp)
    
    new_timestamps = timestamps.copy()
    # print(new_timestamps)
    # print(len(new_timestamps))
    i = 1
    while i < len(new_timestamps):
        if new_timestamps[i] - new_timestamps[i-1] > 2:
            new_timestamps.insert(i, new_timestamps[i-1]+2)
        else:
            i += 1
            
    # print(new_timestamps)
    # print(len(new_timestamps))


    # for i, (measurement_set, timestamp) in enumerate(zip(measurements,timestamps)):
    #     print(f"Timestamp: {timestamp}, Measurment: {measurement_set}")

    ownship = np.zeros((len(new_timestamps),5))

    # measurements_all = np.array([set() for i in timestamps])
    # for i, (measurement_set, timestamp) in enumerate(zip(np.array(measurements), np.array(timestamps))):
    #     for measurement in measurement_set:
    #             measurements_all[i].add(constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp)))
                

    # last_time_stamp = 0
    # counter = 0
    # #measurements_all = np.array([set() for i in timestamps])
    # measurements_all = []
    # for i, (measurement_set, timestamp) in enumerate(zip(np.array(measurements), np.array(timestamps))): 
    #     if timestamp - last_time_stamp > 2:
    #         # new_timestamp = last_time_stamp + 2
    #         measurements_all.append(set())
    #         last_time_stamp +=2
    #         counter += 1
    #     else:
    #         for measurement in measurement_set:
    #             measurements_all.append(set())
    #             measurements_all[-1].add(constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp)))
    #             #measurements_all.append((constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp))))
    #         last_time_stamp = timestamp
    
    # for element in measurements:
    #     for elem in element:
    #         print(elem)
    #         # if not type(elem) == np.ndarray:
    #         #     print(type(elem))
    #         #     print(elem)
    # print(measurements[1][0][0])
    # print(type(measurements[1][0][0]))

    long_i = 0
    short_i = 0
    measurements_all = []
    # print(len(measurements))
    # print(len(measurements[0]))
    # print(len(measurements[0][0]))
    # print((measurements[3]))
    # # for k,elem in enumerate(measurements):
    # #     if len(elem) != 1:
    # #         print(k)
    # # print(np.shape(np.array(measurements)))
    while len(measurements_all) <= len(new_timestamps)-1:
        if new_timestamps[long_i] == timestamps[short_i]:
            for measurement in (measurements)[short_i]:
                #for multiple_measurment in measurement:
                    measurements_all.append(set())
                    measurements_all[-1].add(constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamps[short_i])))
            long_i += 1
            short_i += 1
        else:
            measurements_all.append(set())
            long_i+=1
    measurements_all = np.array(measurements_all)

    # print(measurements_all[26])
    # print(len(measurements_all))
    # print(counter)
    # for elem in measurements_all:
    #     print(elem)

    radar = {1: [constructs.State(ownship_pos, np.identity(4), timestamp) for ownship_pos, timestamp in zip(ownship, new_timestamps)]}

    timestamps = np.array([new_timestamps])
    timestamps = np.reshape(timestamps,(len(timestamps[0]),1))
    # print(measurements_all)
    return measurements_all, radar, timestamps

def radar_data_mat_file(filename):
    mat_data = io.loadmat(filename)

    measurements = []
    timestamps = []
    for i, (timestamp, measurments_set) in enumerate(mat_data.items()):
        # Skip the first three elements in the dictionary, since they are not measurements
        if i<=2: continue
        #print(measurments_set)
        for k,measurement in enumerate(measurments_set):
            measurements.append(set())
            measurements[-1].add(constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp)))
            # if k>0: print(measurements[-1])
            timestamps.append(float(timestamp))


    # I want to check if the exists a timestamp that is not more than 2 seconds from the previous timestamp,
    # if so, I want to insert a new timestamp and a new empty measurment set
    new_timestamps = check_timestamp(timestamps)
    new_measeurements = []
    long_i = 0
    short_i = 0
    while len(new_measeurements) <= len(new_timestamps)-1:
        if new_timestamps[long_i] == timestamps[short_i]:
            new_measeurements.append(measurements[short_i])
            long_i += 1
            short_i += 1
        else:
            new_measeurements.append(set())
            long_i+=1

    measurements = np.array(new_measeurements)
    timestamps = np.array(new_timestamps)
    # measurements = np.array(measurements)
    # timestamps = np.array(timestamps)
    # for (time,measurement) in zip(timestamps,measurements):
    #     print(f"Timestamp: {time}, Measurment: {measurement}")

    
    ownship = np.zeros((len(timestamps),5))
    radar = {1: [constructs.State(ownship_pos, np.identity(4), timestamp) for ownship_pos, timestamp in zip(ownship, timestamps)]}

    timestamps = np.reshape(timestamps,(len(timestamps),1))
    return measurements, radar, timestamps


def check_timestamp(timestamps):
    '''Helper Function for radar_data_mat_file'''
    # I want to check if the exists a timestamp that is not more than 2 seconds from the previous timestamp
    new_timestamps = timestamps.copy()
    #for i in range(len(new_timestamps)-1):
    counter = 0
    i = 0
    while i < len(new_timestamps)-1:
        if new_timestamps[i+1] - new_timestamps[i] > 2:
            #new_timestamps.insert(i+1, new_timestamps[i]+2)
            new_timestamps = np.insert(new_timestamps, i+1, new_timestamps[i]+2)
            #print(f"Inserted new timestamp, {new_timestamps[i]+2}")
            counter += 1
        i += 1
    print(f"Inserted {counter} new timestamps")
    # new_timestamps = timestamps.copy()
    # #for i in range(len(new_timestamps)-1):
    # i = 0
    # while i < len(new_timestamps)-1:
    #     if new_timestamps[i+1] - new_timestamps[i] > 1:
    #         #new_timestamps.insert(i+1, new_timestamps[i]+2)
    #         new_timestamps = np.insert(new_timestamps, i+1, new_timestamps[i]+1)
    #         print(f"Inserted new timestamp, {new_timestamps[i]+2}")
    #     i += 1
    return new_timestamps