from tracking import utilities
from itertools import permutations
import numpy as np

class DataAssociator(object):
    """
    A structure containing a method of computing the marginal association
    probabilities of a cluster.
    """
    def __init__(self):
        pass

    def get_marginal_association_probabilities(self,cluster):
        self.__cluster__ = cluster
        self.__association_hypotheses__ = self.__create_association_hypotheses__()
        self.__association_probabilities__ = self.__compute_association_probabilities__()
        return self.__compute_marginal_association_probabilities__()

class MurtyDataAssociator(DataAssociator):
    """
    A JIPDA style data associator, that uses Murty's Method when the numbers of
    tracks and measurements are large.
    """
    def __init__(self, n_measurements_murty, n_tracks_murty, n_hypotheses_murty):
        self.__n_measurements_murty__ = n_measurements_murty
        self.__n_tracks_murty__ = n_tracks_murty
        self.__n_hypotheses_murty__ = n_hypotheses_murty

    def __compute_marginal_association_probabilities__(self):

        marginal_association_probabilities = np.zeros((self.__cluster__.n_tracks,self.__cluster__.n_measurements+1))
        for k, a_k in enumerate(self.__association_hypotheses__):
            for a_k_i in a_k:
                assigned_measurement = a_k_i[0]
                assigned_track = a_k_i[1]
                marginal_association_probabilities[assigned_track][assigned_measurement] += self.__association_probabilities__[k]
        return marginal_association_probabilities

    def __create_association_hypotheses__(self):

        if self.__cluster__.n_tracks >= self.__n_measurements_murty__ and self.__cluster__.n_measurements >= self.__n_tracks_murty__:
            reward_m = self.__create_reward_matrix__()
            a_all = utilities.murtys_method(reward_m, N=self.__n_hypotheses_murty__)
        else:
            a_all = []
            measurement_indices = list(range(self.__cluster__.n_measurements))
            measurement_indices.extend([self.__cluster__.n_measurements]*self.__cluster__.n_tracks)
            measurement_permutations = list(set(list(permutations(measurement_indices,self.__cluster__.n_tracks))))
            for permutation in measurement_permutations:
                a = []
                for track_index, measurement_index in enumerate(permutation):
                    a.append([measurement_index, track_index])
                a_all.append(a)
        return a_all

    def __compute_association_probabilities__(self):

        association_probabilities = np.zeros(len(self.__association_hypotheses__))
        for k, a_k in enumerate(self.__association_hypotheses__):
            log_sum = 0
            for i, a_k_i in enumerate(a_k):
                if a_k_i[0] < self.__cluster__.n_measurements:
                    log_sum +=  np.log(self.__cluster__.w[a_k_i[1],a_k_i[0]]/self.__cluster__.clutter_density)
                else:
                    log_sum +=  np.log(self.__cluster__.w[a_k_i[1],a_k_i[0]])
            association_probabilities[k] = np.exp(log_sum)
        association_probabilities = association_probabilities/np.sum(association_probabilities)
        return association_probabilities

    def __create_reward_matrix__(self):
        """
        Creates a reward matrix for use in Murty's method, with the tracks and
        measurements in the cluster.
        """
        reward_m = np.zeros((self.__cluster__.n_measurements+self.__cluster__.n_tracks,self.__cluster__.n_tracks), dtype=[('value','float'),('index','2int8')])
        for i, estimate in enumerate(self.__cluster__.tracks):
            for j, measurement in enumerate(self.__cluster__.measurements):
                if measurement in estimate.measurements:
                    reward_m['value'][j,i] = np.log(self.__cluster__.w[i,j]) - np.log(self.__cluster__.clutter_density)
                    reward_m['index'][j,i] = [j,i]
                else:
                    reward_m['value'][j,i] = -1e100
                    reward_m['index'][j,i] = [j,i]
            for j in range(self.__cluster__.n_tracks):
              reward_m['value'][self.__cluster__.n_measurements+j,i] = -1e100
              reward_m['index'][self.__cluster__.n_measurements+j,i] = [self.__cluster__.n_measurements+j,i]
            reward_m['value'][self.__cluster__.n_measurements+i,i] = np.log(self.__cluster__.w[i,-1])
            reward_m['index'][self.__cluster__.n_measurements+i,i] = [self.__cluster__.n_measurements+i,i]
        return reward_m
