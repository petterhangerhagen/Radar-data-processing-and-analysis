from itertools import combinations
import numpy as np
from scipy.stats import chi2

class Terminator():
    def __init__(self, termination_threshold, max_steps_without_measurements=8, fusion_significance_level=0.1):
        self.__termination_threshold__ = termination_threshold
        self.__max_steps_without_measurements__ = max_steps_without_measurements
        self.__tracks_without_measurements__ = dict()
        self.__fuse_threshold__ = chi2(df=2).ppf(1-fusion_significance_level)

    def step(self, tracks):
        dead_tracks = set()
        for track in tracks:
            # chack if any tracks have existence probability below the threshold
            if self.__termination_check__(track):
                dead_tracks.add(track)

            # chack and update if any tracks haven't associated to a measurement
            if self.__no_measurement_termination_check__(track):
                dead_tracks.add(track)

        # check if two tracks are the same, for all combinations of tracks
        for track1, track2 in combinations(tracks, 2):
            if self.__terminate_identical__(track1, track2):
                dead_tracks.add(max([track1,track2], key=lambda x:x.index))

        return tracks - dead_tracks

    def __termination_check__(self, track):
        return track.existence_probability < self.__termination_threshold__

    def __no_measurement_termination_check__(self, track):
        index = getattr(track,'index')
        if index in self.__tracks_without_measurements__.keys():
            if len(track.measurements) == 0:
                self.__tracks_without_measurements__[index] += 1
            else:
                self.__tracks_without_measurements__[index] = 0
        else:
            if len(track.measurements) == 0:
                self.__tracks_without_measurements__[index] = 1
            else:
                self.__tracks_without_measurements__[index] = 0

        return self.__tracks_without_measurements__[index] >= self.__max_steps_without_measurements__

    def __terminate_identical__(self, track1, track2):
        """
        Testing the hypothesis that the two tracks are the same. Based on the
        Method from Multitarget-multisensor tracking: Principles and techniques
        chapter 8
        """
        n_x = 4
        rho = 0.4
        mean1, covariance1 = track1.posterior
        mean2, covariance2 = track2.posterior
        P_tj = np.multiply(rho, covariance1[0:4,0:4])

        delta = (mean1[0:4]-mean2[0:4]).reshape((n_x,1))

        T = covariance1[0:4,0:4]+covariance2[0:4,0:4]-P_tj-P_tj.T

        D = delta.T.dot(np.linalg.inv(T).dot(delta)).squeeze()
        return D < self.__fuse_threshold__
