import numpy as np
from copy import deepcopy
from tracking import utilities


class Tracker():
    """
    Generic tracker class
    """
    def __init__(self, filter):
        self.filter = filter


class VIMMJIPDATracker(Tracker):
    """
    The VIMMJIPDA tracker. It can be summarized as a JIPDA with mulitple models
    and modeling of the visibility of the tracks.
    """
    def __init__(self, filter, clutter_model, data_associator, detection_probability, survival_probability, visibility_transition_matrix, gamma = 3, single_target = False, visibility_off = False):
        super().__init__(filter)
        self.clutter_model = clutter_model
        self.detection_probability = detection_probability
        self.visibility_transition_matrix = visibility_transition_matrix
        self.survival_probability = survival_probability
        self.gamma = gamma
        self.data_associator = data_associator
        self.single_target = single_target   # only for testing/comparison
        self.visibility_off = visibility_off # only for testing/comparison


    def step(self, previous_tracks, measurements, timestamp):
        # Prediction
        tracks = {self.predict(previous_track, timestamp) for previous_track in previous_tracks}

        # Gate tracks
        tracks = {utilities.gate_track(track, measurements, self.gamma) for track in tracks}

        # Clustering
        if self.single_target:
            clusters, unused_measurements = utilities.one_track_clustering(tracks,measurements)
        else:
            clusters, unused_measurements = utilities.single_linkage_clustering(tracks,measurements)

        for cluster in clusters:
            # Initialize cluster
            cluster = self.initialize_cluster(cluster)

            # Calculate posterior values conditional on measurements
            clusters = self.measurement_conditional_posteriors(cluster)

            # Mixture reduction
            self.mixture_reduction(cluster)
        return tracks, unused_measurements

    def predict(self, previous_track, timestamp):
        track = deepcopy(previous_track)
        track.timestamp = timestamp

        dt = timestamp-previous_track.timestamp

        self.filter.predict(track, dt)

        track.existence_probability = self.predict_existence_probability(track)

        track.visibility_probability = self.predict_visibility_probability(track)
        return track



    def initialize_cluster(self, cluster):
        cluster.n_z ,cluster.n_x = self.filter.measurement_model.get_measurement_mapping().shape
        cluster.n_tracks = len(cluster.tracks)
        cluster.n_modes = len(cluster.tracks[0].kinematic_models)
        cluster.n_measurements = len(cluster.measurements)
        cluster.clutter_density = self.clutter_model.get_clutter_density()
        cluster.innovation, cluster.S, cluster.S_inv = cluster.calculate_innovation()
        cluster.measurement_likelihoods = cluster.calculate_measurement_likelihoods()
        return cluster

    def measurement_conditional_posteriors(self, cluster):
        M = cluster.n_modes
        m_k = cluster.n_measurements
        n_t = cluster.n_tracks
        P_D = self.detection_probability


        cluster.r = np.zeros((n_t,m_k+1))
        cluster.eta = np.zeros((n_t,m_k+1))
        cluster.mu = np.zeros((n_t,M,m_k+1))
        cluster.w = np.zeros((n_t,m_k+1))

        for t, track in enumerate(cluster.tracks):

            measurement_likelihoods = cluster.measurement_likelihoods[t]
            measurement_likelihoods_combined = np.dot(track.mode_probabilities,cluster.measurement_likelihoods[t])

            mode_probabilities = track.mode_probabilities

            existence_prob_miss = ((1-P_D*track.visibility_probability)*track.existence_probability)/(1-P_D*track.visibility_probability*track.existence_probability)
            visibility_prob_miss = (1-P_D)*track.visibility_probability/(1-P_D*track.visibility_probability)

            cluster.r[t] = np.hstack((np.ones(m_k),existence_prob_miss)).reshape(m_k+1)
            cluster.eta[t] = np.hstack((np.ones(m_k),visibility_prob_miss)).reshape(m_k+1)
            cluster.mu[t] = np.concatenate((np.exp(np.log(mode_probabilities.reshape(M,1))+np.log(measurement_likelihoods.reshape(M,m_k)) - \
                np.log(measurement_likelihoods_combined.reshape(1,m_k))),(mode_probabilities.reshape(M,1).reshape(M,1))),axis=1).reshape((M,m_k+1))
            cluster.w[t] = np.hstack((P_D*track.visibility_probability*track.existence_probability*measurement_likelihoods_combined,\
                1-P_D*track.visibility_probability*track.existence_probability)).reshape((m_k+1))
        return cluster



    def mixture_reduction(self, cluster):
        n_x = cluster.n_x
        n_z = cluster.n_z

        M = cluster.n_modes
        m_k = cluster.n_measurements
        n_t = cluster.n_tracks

        marginal_association_probabilities = self.data_associator.get_marginal_association_probabilities(cluster)

        for t, track in enumerate(cluster.tracks):
            eta_t_j = cluster.eta[t]
            r_t_j = cluster.r[t]
            mu_t_s_j = cluster.mu[t]
            p_t_j = marginal_association_probabilities[t]

            track.existence_probability = np.dot(r_t_j.T,p_t_j).squeeze()

            if self.visibility_off:
                track.visibility_probability = 1
            else:
                track.visibility_probability = np.exp(np.log(np.sum(np.exp(np.log(eta_t_j.reshape(m_k+1)) + \
                    np.ma.log(p_t_j.reshape(m_k+1)) + np.log(r_t_j.reshape(m_k+1)))))-(np.log(track.existence_probability)))


            track.mode_probabilities = np.exp(np.log(np.sum(np.exp(np.log(mu_t_s_j) + np.ma.log(p_t_j.reshape(1,m_k+1)) + \
                np.log(r_t_j.reshape(1,m_k+1))),axis=1))-(np.log(track.existence_probability))).reshape(M)


            betas = np.exp(np.log(mu_t_s_j) + np.ma.log(p_t_j.reshape(1,m_k+1)) + np.log(r_t_j.reshape(1,m_k+1))- \
                (np.log(track.mode_probabilities.reshape(M,1)) + np.log(track.existence_probability))).filled(0).reshape(M,m_k+1)


            self.filter.update(
                track,
                cluster.innovation[t],
                betas
            )

            track.mode_probabilities = utilities.ensure_min_mode_probability(0.001, track.mode_probabilities)



    def predict_visibility_probability(self, track):
        if self.visibility_off:
            return 1
        return self.visibility_transition_matrix[0,0]*track.visibility_probability+self.visibility_transition_matrix[1,0]*(1-track.visibility_probability)

    def predict_existence_probability(self, track):
        # Tidsuavhengig. Legge inn tomme målinger 
        return self.survival_probability*track.existence_probability
