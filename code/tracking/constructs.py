from tracking import models
from tracking import utilities
import numpy as np
import collections
import anytree
import copy


class Cluster(object):
    """
    A cluster of tracks. Is used to perform basic calculations on the tracks and
    measurements in the cluster, and as a tool to pass relevant information
    between functions in a tracker.
    """
    def __init__(self, tracks, measurements):
        self.tracks = tracks
        self.measurements = measurements

    def add_track(self, track):
        self.tracks.append(track)

    def add_measurements(self, measurements):
        self.measurements = self.measurements | measurements


    def calculate_innovation(self):
        """
        Calculates the innovation between the tracks and measurements in the cluster.
        """

        S = []
        z_hat = []
        S_inv = []
        for t, track in enumerate(self.tracks):
            track_shape = track.states.shape
            z_hat_t, S_t = track.states.get_mean_covariance_array(track.predicted_measurements)
            z_hat.append(z_hat_t.reshape(track_shape + (1, self.n_z)))
            S.append(S_t.reshape(track_shape + (self.n_z, self.n_z)))
            S_inv.append(np.linalg.inv(S_t).reshape(track_shape + (self.n_z, self.n_z)))


        if self.measurements == set():
            innovation = [None]*self.n_tracks
        else:
            innovation = []
            measurement_values = np.array([measurement.value.reshape((self.n_z)) for measurement in self.measurements])
            for t, track in enumerate(self.tracks):
                innovation.append(measurement_values-z_hat[t])
        return innovation, S, S_inv

    def calculate_measurement_likelihoods(self):
        """
        Calculates the measurement likelihoods between the tracks and measurements
        in the cluster.
        """

        measurement_likelihoods = []
        for t, estimate in enumerate(self.tracks):
            measurement_likelihoods_t = np.zeros(estimate.states.shape + (self.n_measurements,))
            innovation = self.innovation[t]
            S_inv = self.S_inv[t]
            determinant = np.linalg.det(self.S[t])
            for j, measurement in enumerate(self.measurements):

                a = np.matmul(innovation[...,j,None,:], S_inv[...,:,:])[...]

                b = innovation[...,j,:,None]
                exponent = np.matmul(a,b)[...,0,0]
                measurement_likelihoods_t[...,j] = np.divide(np.exp(-0.5*exponent),(2*np.pi*np.sqrt(determinant)))



            # to avoid division by zero in the case of large validation gates
            measurement_likelihoods_t[measurement_likelihoods_t < 1e-100] = 1e-100
            measurement_likelihoods.append(measurement_likelihoods_t)

        return measurement_likelihoods



class Track(object):
    """
    Constains all information regarding the individual tracks. Only states and
    probabilities defined in __allowed_states__ and __allowed_probabilities__
    can be collected from **kwargs.

    The states (which can be more than one when using a hybrid state framework)
    are defined as a tree structure (Consisting of TrackState instances). The
    one pointed to by self.states is the root of the tree structure.
    """
    def __init__(self, timestamp, states, index, existence_probability, **kwargs):
        self.__allowed_states__ = {'kinematic_models'}
        self.__allowed_probabilities__ = {'visibility_probability', 'mode_probabilities'}
        self.__state_probability_combination__ = {'kinematic_models': 'mode_probabilities'}
        self.timestamp = timestamp
        self.index = index
        self.measurements = set()
        self.existence_probability = existence_probability
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__allowed_states__)
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__allowed_probabilities__)
        self.states = states

    def __eq__(self, other):
        return (self.index == getattr(other, 'index'))

    def __hash__(self):
        return hash((self.index, self.timestamp))

    def __repr__(self):
        mean, cov = self.posterior
        return f'Track {self.index}: [{mean[0]:5.2f}, {mean[2]:5.2f}], r = {self.existence_probability:.3f}'


    """
    Makes it possible to set the porbabilities of the kinematic models without
    having to access the tree-structured states from outside. E.g., by calling
    track.mode_probabilities = new_probabilities the node of the tree holding
    the mode-probabilities is updated.
    """
    def __setattr__(self, property, value):
        if '__state_probability_combination__' in self.__dict__:
            if property in self.__state_probability_combination__.values():
                level_grouped_nodes = anytree.LevelOrderGroupIter(self.states)
                for level in level_grouped_nodes:
                    if property in level[0].__dict__:
                        if len(level) > 1:
                            for i, node in enumerate(level):
                                assert(getattr(node, property).shape == value[i].shape)
                                setattr(node ,property ,value[i])
                        else:
                            setattr(level[0] ,property ,value)
                        return
            else:
                # default behavior
                return object.__setattr__(self, property, value)
        else:
            # default behavior
            return object.__setattr__(self, property, value)

    """
    Similar to __setattr__, but returns the wanted values instead. E.g. with
    new_probabilities = track.mode_probabilities the mode probabilities are
    returned.
    """
    def __getattribute__(self, property):
        if '__state_probability_combination__' in object.__getattribute__(self, '__dict__'):
            if property in object.__getattribute__(self, '__state_probability_combination__').values():
                level_grouped_nodes = anytree.LevelOrderGroupIter(self.states)
                for level in level_grouped_nodes:
                    if property in level[0].__dict__:
                        if len(level) > 1:
                            property_list = []
                            for node in level:
                                property_list.append(getattr(node, property))
                            return np.asarray(property_list)
                        else:
                            return np.asarray(getattr(level[0], property))
            else:
                # default behavior
                return object.__getattribute__(self, property)
        else:
            # default behavior
            return object.__getattribute__(self, property)


    @property
    def posterior(self):
        mean, covariance = self.__get_single_posterior__(self.states)
        return mean.squeeze(), covariance.squeeze()

    """
    Returns a single posterior, i.e. a Gaussian mixture of all the states.
    """
    def __get_single_posterior__(self, node):
        if node.children:
            for state, probabilities in node.__dict__.items():
                if state in node.__allowed_probabilities__:
                    means = []
                    covariances = []
                    for i, child in enumerate(node.children):
                        mean, covariance = self.__get_single_posterior__(child)
                        means.append(mean)
                        covariances.append(covariance)
                    means = np.asarray(means).reshape(len(node.children), means[0].shape[0])
                    covariances = np.asarray(covariances).reshape(len(node.children), covariances[0].shape[-1], covariances[0].shape[-1])
                    return utilities.gaussian_mixture(means, covariances, probabilities)
        else:
            return node.mean, node.covariance



class State(object):
    """
    A simple kinematic pdf, which can be equipped with a timestamp and an id.
    """
    def __init__(self, mean, covariance, timestamp=None, id=None):
        self.mean = mean
        self.covariance = covariance
        self.timestamp = timestamp
        self.id = id

    def update(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    @property
    def posterior(self):
        return self.mean, self.covariance
    
    def __repr__(self):
        return f'State: [{self.mean[0]}, {self.mean[1]}]'

class TrackState(State, anytree.NodeMixin):
    """
    A node in a tree of states, belonging to a track. The states with kinematic
    pdfs are all leaf nodes of the tree. All non-leaf nodes hold the probability
    of each of their children. For the case where three kinematic models are
    used, and the kinematic pdfs are only conditoned on these, the tree would
    consist of four nodes:

    A      = Node holding the mode probabilities
    |– CV1 = Node holding the kinematic pdf conditioned on mode 1
    |– CT  = Node holding the kinematic pdf conditioned on mode 2
    |– CV2 = Node holding the kinematic pdf conditioned on mode 3

    The 'name' state of each node consists its defining characteristic, which
    for the mode-conditioned pdfs are their kinematic models.
    """
    def __init__(self, mean=None, covariance=None, id=None, parent=None, children=None, name=None, **kwargs):
        super().__init__(mean, covariance, id)
        self.__allowed_probabilities__ = {'visibility_probability', 'mode_probabilities', 'mmsi_probabilities'}
        self.parent = parent
        self.name = name if parent is not None else None
        if children:
            self.children = children
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__allowed_probabilities__)

    """
    Translates the tree structure to a numpy array of the means and covariances,
    with the same shape as the tree.
    """
    def get_mean_covariance_array(self, node=None):
        if node is None:
            node = self
        means = []
        covariances = []
        for i, leaf_state in enumerate(node.leaves):
            means.append(np.asarray(leaf_state.mean))
            covariances.append(np.asarray(leaf_state.covariance))
        means = np.asarray(means).reshape(node.shape + (means[0].shape[0],))
        covariances = np.asarray(covariances).reshape(self.shape + (covariances[0].shape[-1], covariances[0].shape[-1]))

        return means, covariances

    def update_from_mean_covariance_array(self, means, covariances):
        for s, leaf_state in enumerate(self.leaves):
            leaf_state.update(means[s], covariances[s])

    """
    The shape of a node is in this context defined as (the number of children)x
    (the number of children for each children) etc. The structure of the tree
    will always be such that all nodes on the same level has the same number of
    children.
    """
    @property
    def shape(self):
        node = self.children[0]
        shape = ()
        while True:
            shape += (len(node.siblings)+1,)
            if node.height == 0:
                break
            node = node.children[0]
        return shape

    """
    Defining the len()-function to return the number of kinematic pdfs.
    """
    def __len__(self):
        return len(self.leaves)



class Measurement(State):
    def __init__(self, value, covariance, timestamp, id=None):
        super().__init__(mean=value, covariance=covariance, timestamp=timestamp, id=id)

    def __eq__(self, other):
        return (self.value[0] == other.value[0] and self.value[1] == other.value[1] and self.timestamp == other.timestamp)

    def __hash__(self):
        return hash((self.mean[0], self.mean[1], self.timestamp))

    def __repr__(self):
        return f'Measurement: [{self.mean[0]}, {self.mean[1]}]'

    @property
    def value(self):
        return self.mean
