from tracking import utilities, constructs
from tracking.models import KinematicModel
import numpy as np
import anytree
import copy

class StateFilter():
    """
    Generic Filter class.
    """
    def __init__(self):
        pass

class KalmanFilter(StateFilter):
    """
    A regular Kalman filter class.
    """
    def __init__(self, measurement_model):
        self.measurement_model = measurement_model

    def predict(self, track, dt):
        # loop through and predict all states
        for state in track.states.leaves:
            predicted_mean, predicted_covariance = state.name.step(state, dt)
            state.update(predicted_mean, predicted_covariance)

        # initialize the predicted measurements of each state
        track.predicted_measurements = self.__get_predicted_measurements__(track.states)
        return track

    """
    Returns the predicted measurements on the same form as the tree of states.
    """
    def __get_predicted_measurements__(self, states):
        predicted_measurements = copy.deepcopy(states)
        for predicted_measurement in predicted_measurements.leaves:
            z_hat, S = self.measurement_model.predict_measurement(predicted_measurement)
            predicted_measurement.update(z_hat, S)
        return predicted_measurements

    def update(self, track, innovation, weights):
        if innovation is None:
            return
        predictions = track.predicted_measurements
        innovation.reshape(len(track.states),innovation.shape[-2],innovation.shape[-1])
        weights.reshape(len(track.states),innovation.shape[-2]+1)

        # loop through and update all states
        for i, (state, prediction) in enumerate(zip(track.states.leaves, predictions.leaves)):
            self.__update_state__(state, prediction, innovation[i], weights[i])

    """
    Updating of an individual state, i.e. the update step of the Kalman filter.
    """
    def __update_state__(self, state, pred, innovation_all, weights):
        covariance = state.covariance
        mean = state.mean
        S = pred.covariance
        S_inv = np.linalg.inv(S)
        H = self.measurement_model.get_measurement_mapping()
        n_z = H.shape[0]
        n_x = H.shape[1]

        kalman_gain = np.dot(covariance,np.dot(H.T, S_inv)).reshape((n_x, n_z))
        total_innovation = np.zeros((n_z, 1))
        cov_terms = np.zeros((n_z, n_z))
        for innovation, weight in zip(innovation_all, weights[:-1]):
            total_innovation += weight*innovation.reshape(n_z,1)
            innovation_vec = innovation.reshape((n_z, 1))
            cov_terms += weight*innovation_vec.dot(innovation_vec.T)
        cov_terms -= total_innovation.dot(total_innovation.T)
        soi = np.dot(kalman_gain, np.dot(cov_terms, kalman_gain.T))
        P_c = np.dot(kalman_gain, np.dot(S, kalman_gain.T))
        mean = mean+kalman_gain.dot(total_innovation).reshape(n_x)
        covariance = covariance-(1-weights[-1])*P_c+soi
        state.update(mean, covariance)

class IMMFilter(KalmanFilter):
    """
    An Extended Kalman Filter class working with several kinematic models.
    """
    def __init__(self, measurement_model, mode_transition_matrix):
        super().__init__(measurement_model)
        self.__mode_transition_matrix__ = mode_transition_matrix

    def predict(self, track, dt):
        """
        NOTE: here it is assumed that the modes are conditioned on all other
        eventual discrete states in the hybrid state. To make it more general
        one will have to search the state tree for the nodes holding the mode
        probabilities.
        """
        # get the prior mode-conditional states
        for state in anytree.findall_by_attr(track.states, 1, name='height'):
            self.__IMM_mix__(state)

        # loop through and predict all states
        for state in track.states.leaves:
            predicted_mean, predicted_covariance = state.name.step(state, dt)
            state.update(predicted_mean, predicted_covariance)


        track.predicted_measurements = self.__get_predicted_measurements__(track.states)
        return track

    def __IMM_mix__(self, state):
        mode_probabilities = state.mode_probabilities
        joint_mode_probabilities = np.transpose(np.multiply(self.__mode_transition_matrix__.T,mode_probabilities))
        prior_mode_probabilities = np.sum(joint_mode_probabilities,axis=0)
        conditional_mode_probabilities = np.divide(joint_mode_probabilities,prior_mode_probabilities)
        state.mode_probabilities = prior_mode_probabilities
        means, covariances = state.get_mean_covariance_array()
        means, covariances = utilities.gaussian_mixture(means, covariances, conditional_mode_probabilities)
        state.update_from_mean_covariance_array(means, covariances)
