import numpy as np
from tracking import constructs
import operator


def gate_track(track, measurements, gamma):
    """
    Adds all measurements which fall within a track's validation gate to the
    set of measurements contained by the track.
    """
    measurements_used = set()
    for state in track.predicted_measurements.leaves:
        z_hat = state.mean
        S = state.covariance
        S_inv = np.linalg.inv(S)
        for measurement in measurements:
            z = measurement.value
            nu = z - z_hat
            if nu.T.dot(S_inv).dot(nu) < gamma:
                measurements_used.add(measurement)
    track.measurements = measurements_used
    return track


def single_linkage_clustering(tracks, measurements):
    """
    Place all tracks that share measurements in the same cluster.
    """
    used_measurements = set()
    clusters = []

    for track in tracks:
        gated_measurements = track.measurements

        if used_measurements & gated_measurements != set():
            for cluster in clusters:
                if cluster.measurements & gated_measurements != set():
                    cluster.add_track(track)
                    cluster.add_measurements(gated_measurements)
                    break
        else:
            cluster = constructs.Cluster([track],gated_measurements)
            clusters.append(cluster)
        used_measurements = used_measurements | gated_measurements
    unused_measurements = measurements - used_measurements
    return clusters, unused_measurements

def one_track_clustering(tracks, measurements):
    """
    Place each track in its own cluster, giving single target tracking behavior.
    """
    used_measurements = set()
    clusters = []

    for track in tracks:
        gated_measurements = track.measurements
        cluster = constructs.Cluster([track],gated_measurements)
        clusters.append(cluster)
        used_measurements = used_measurements | gated_measurements
    unused_measurements = measurements - used_measurements
    return clusters, unused_measurements

def gaussian_mixture(means, covariances, weights):
    """
    Returns the mixture of Gaussians. Works with arrays of weights on the form
    1xM (returns single Gaussian) and MxM (returns M Gaussians).
    """
    outer_product = np.array([np.outer(mean,mean) for mean in means])
    covariances = np.dot(weights.T,covariances.reshape(covariances.shape[0], covariances.shape[1]**2)\
     + outer_product.reshape(covariances.shape[0],covariances.shape[1]**2)).reshape(np.atleast_2d(weights).shape[0], covariances.shape[1], covariances.shape[2])
    new_means = np.dot(weights.T,means)

    new_covariances = covariances - np.array([np.outer(mean,mean) for mean in np.atleast_2d(new_means)])

    return new_means, new_covariances



def ensure_min_mode_probability(min_mode_probability, mode_probabilities, tau=None):
    """
    Ensures that no mode probabilities are smaller than min_mode_probability.
    """
    n_modes = len(mode_probabilities)
    if any(mode_probability < min_mode_probability for mode_probability in mode_probabilities):
        indices = [idx for idx, mode_probability in enumerate(mode_probabilities) if mode_probability < min_mode_probability]
        subtract_value = sum(np.array([min_mode_probability]*len(indices)) - np.array(mode_probabilities[indices]))
        indices_inv = [i for i in range(n_modes) if i not in indices]
        mode_probabilities[indices] = min_mode_probability
        if any(mode_probability-subtract_value/(n_modes-len(indices)) < min_mode_probability for mode_probability in mode_probabilities[indices_inv]):
            max_index = np.argmax(mode_probabilities)
            min_index = np.argmin(mode_probabilities)
            subtract_value -= mode_probabilities[min_index]-min_mode_probability
            mode_probabilities[min_index] = min_mode_probability
            mode_probabilities[max_index] -= subtract_value
        else:
            for i in indices_inv:
                mode_probabilities[i] = mode_probabilities[i]-subtract_value
    return mode_probabilities

def auction_algorithm(reward_m, epsilon = 0.01):
    """
    Returns the best assignment between a number of participants and items. The
    reward each item j gives participant i is found in the (j,i)-element of the
    (n_participants+n_items)x(n_participants) input matrix.
    """

    if reward_m.size == 1:
        # only one possible assignment
        return [[0,0]], reward_m['value'][[0,0]]

    # initialize constants and arrays
    n_participants = reward_m.shape[1]
    n_items = reward_m.shape[0]-n_participants
    reward = 0
    unassigned_participants = []
    assigned_participants = []
    for i in range(n_participants):
        unassigned_participants.append([-1,i])
    prices = np.zeros(n_items+n_participants)
    assigned_items = []


    # run until all participants have been assigned an item
    while unassigned_participants != []:
        # remove the first of the unnasigned participants
        selected_participant = unassigned_participants.pop(0)

        # assign items to remaining participants
        for participant in unassigned_participants:
            preferred_item = np.argmax(reward_m['value'][:,participant[1]]-prices)
            if preferred_item not in assigned_items:
                # if item is available, assign to participant
                participant[0] = preferred_item
                assigned_items.append(preferred_item)
                unassigned_participants.remove(participant)
                assigned_participants.append(participant)

        # assign item to the selected participant
        preferred_item = np.argmax(reward_m['value'][:,selected_participant[1]]-prices)
        selected_participant[0] = preferred_item

        # if the selected participant's item is held by another participant, take it from the other participant
        if preferred_item in assigned_items:
            for participant in assigned_participants:
                if participant[0] == preferred_item:
                    assigned_participants.remove(participant)
                    unassigned_participants.append(participant)
        else:
            assigned_items.append(preferred_item)
        assigned_participants.append(selected_participant)

        # the price of the selected participant's item is increased according to the difference between the best and second best item
        values = reward_m['value'][:,selected_participant[1]] - prices
        values = np.sort(values)
        prices[selected_participant[0]] = prices[selected_participant[0]] + (values[-1]-values[-2]) + epsilon # epsilon ensures convergence

    for i in range(n_participants):
        reward = reward + reward_m['value'][assigned_participants[i][0],assigned_participants[i][1]] # compute combined reward
        assigned_participants[i] = reward_m['index'][assigned_participants[i][0],assigned_participants[i][1]] # get assignment value from reward matrix

    assigned = np.asarray(assigned_participants)
    assigned = assigned[np.argsort(assigned[:,0])]

    return assigned, reward

def murtys_method(reward_m, N=10, epsilon=0.01):
    """
    Takes in a reward matrix and returns the N best association hypotheses
    according to the matrix.
    """
    a_i, _ = auction_algorithm(reward_m) # find the single best hypothesis

    a = [a_i] # initialize the list of N best hypotheses
    i = 0
    reward_m_temp = reward_m.copy()
    L = [] # list for storing the solutions

    # run loop N times to get the N best hypotheses
    while i < N:
        j = reward_m.shape[1]-reward_m_temp.shape[1]
        while True:
            # make the j-th assignment in the previous best hypothesis unavailable
            unassignable_idx = np.where( np.all(reward_m_temp['index'] == np.array(a_i[j]), axis=-1) )
            reward_m_temp['value'][unassignable_idx] = -1e100
            # find the best hypothesis under the new problem
            a_new, reward_new = auction_algorithm(reward_m_temp)

            if j > 0:
                # the values of the deleted rows and columns are the same as for a_i, i.e. the best solution for the previous problem
                a_new = np.concatenate((a_i[0:j],a_new))
                for t in a_i[0:j]:
                     reward_new = reward_new + reward_m['value'][t[0],t[1]]

            L.append((reward_m_temp, a_new, reward_new))
            if reward_m_temp.shape[1] == 1:
                # no more solutions can be found
                break

            reward_m_temp = np.delete(reward_m_temp,unassignable_idx[0],axis=0)
            reward_m_temp = np.delete(reward_m_temp,unassignable_idx[1],axis=1)
            j += 1


        L.sort(key=operator.itemgetter(2)) # sort L according to the rewards
        a_i = L[-1][1] # get the best hypothesis in L
        reward_m_temp = L[-1][0] # the new reward matrix is the one belonging to the best solution in L
        L.pop(-1) # remove the best solution from L
        a.append(a_i)

        i += 1

    # convert the association hypotheses to the correct format
    for k, a_k in enumerate(a):
        a[k] = list(a_k)
        for p, a_p in enumerate(a_k):
            a[k][p] = list(a_p)
            if a[k][p][0] >= reward_m.shape[0]-reward_m.shape[1]:
                a[k][p][0] = reward_m.shape[0]-reward_m.shape[1]
    return a
