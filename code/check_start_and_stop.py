
## Author: Petter Hangerhagen ##
## Date: 17.11.2023 ##

"""
Description:
"""

import numpy as np

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


class CountMatrix:
    def __init__(self,reset=False):
        self.file_name = "/home/aflaptop/Documents/radar_tracker/data/count_matrix.npy"
        self.count_matrix = np.load(self.file_name)
        if reset:
            self.reset_count_matrix()
        
    def check_start_and_stop(self,track_history):
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

        np.save(self.file_name, self.count_matrix)


    def reset_count_matrix(self):
        zero_matrix = np.zeros((6,6))
        print("Resetting matrix")
        np.save(self.file_name,zero_matrix)