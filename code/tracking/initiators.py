import numpy as np
import tracking.models as models
from tracking.constructs import State, Track, TrackState
from scipy.linalg import block_diag
import copy
import anytree


class Initiator():
    def __init__(self, initial_existence_probability, measurement_model, initial_vel_cov, **other_initial_values):
        self.__allowed_initial_states__ = ['kinematic_models']
        self.__allowed_initial_probabilities__ = ['mode_probabilities', 'visibility_probability']
        self.__state_probability_combination__ = {'kinematic_models': 'mode_probabilities'}
        self.__independent_probabilities__ = ['visibility_probability']
        self.__initial_existence_probability__ = initial_existence_probability
        self.__initial_vel_cov__ = initial_vel_cov
        self.__measurement_model__ = measurement_model
        self.__dict__.update((k, v) for k, v in other_initial_values.items() if k in self.__allowed_initial_states__)
        self.__dict__.update((k, v) for k, v in other_initial_values.items() if k in self.__allowed_initial_probabilities__)
        self.__index_count__ = 1


class SinglePointInitiator(Initiator):
    def step(self, unused_measurements, **kwargs):
        new_tracks = set()

        # initiate tracks on measurements
        for measurement in unused_measurements:
            # calculate the covariance measurement
            measurement_covariance = self.__measurement_model__.get_measurement_covariance(measurement.value)

            # initialize a track on the measurements
            track = self.__initiate_track__(measurement, measurement_covariance=measurement_covariance)

            # add track to the set of new tracks
            new_tracks.add(track)

            # change the index count, so the next new track gets a different index
            self.__index_count__ += 1
        return new_tracks

    def __initiate_track__(self, measurement, measurement_covariance=None):
        mean = measurement.value
        if measurement_covariance is None:
            R = measurement.covariance
        else:
            R = measurement_covariance
        mean = np.hstack((mean, np.zeros(3)))
        covariance = block_diag(R, self.__initial_vel_cov__*np.identity(3))
        mapping = np.array([[1, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 0]])

        mean = mapping.dot(mean)
        covariance = mapping.dot(covariance).dot(mapping.T)
        state = State(mean, covariance)
        state_dict = self.__initialize_states__(state)
        other_states_and_probabilities = dict()
        for k, v in self.__dict__.items():
            if k in self.__allowed_initial_states__ or k in self.__allowed_initial_probabilities__:
                other_states_and_probabilities[k] = v

        # if no kinematic models are specified, a default CV model is chosen.
        if 'kinematic_models' not in other_states_and_probabilities.keys():
            other_states_and_probabilities['kinematic_models'] = models.CVModel(0.1)

        kwargs = {k:v for k, v in self.__dict__.items() if k in self.__independent_probabilities__ or k in self.__allowed_initial_states__}
        new_track = Track(
            measurement.timestamp,
            state_dict,
            self.__index_count__,
            self.__initial_existence_probability__,
            measurements = {measurement},
            **kwargs
        )
        return new_track

    def __initialize_states__(self, state):
        """
        Initialization of the tree containing the states of the hybrid state.
        """
        states = self.__set_state__(state, None, None, self.__allowed_initial_states__)
        return states

    def __set_state__(self, state, this_discrete_state, root, allowed_initial_states):
        """
        Recursively creates the nodes of the tree. The leaf nodes contain the
        states (or kinematic pdfs), while all parents hold the probabilities of
        their children.
        """
        kwargs = dict()
        for discrete_state in allowed_initial_states:
            if discrete_state in self.__dict__:
                key = self.__state_probability_combination__[discrete_state]
                value = self.__dict__[key]
                kwargs[key] = value
                break

        state_node = TrackState(state.mean, state.covariance, parent = root, name=this_discrete_state, children = None, **kwargs)
        next_allowed_initial_states = copy.deepcopy(allowed_initial_states)
        for discrete_state in allowed_initial_states:
            if discrete_state in self.__dict__:
                next_allowed_initial_states.remove(discrete_state)
                state_node.children = [self.__set_state__(state, d_state, state_node, next_allowed_initial_states) for d_state in self.__dict__[discrete_state]]
                return state_node
            else:
                next_allowed_initial_states.remove(discrete_state)
        return state_node
