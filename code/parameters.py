import numpy as np

tracker_state = dict()
tracker_state["IMM_off"] = True
tracker_state["single_target"] = False
tracker_state["visibility_off"] = False

tracker_params = dict()
tracker_params['maximum_velocity'] = 0 # Not used
tracker_params['init_Pvel'] = 3**2
tracker_params['P_D'] = 0.92
tracker_params['clutter_density'] = 1e-5
tracker_params['gamma'] = 3.0**2
tracker_params['survival_prob'] = 0.9   
tracker_params['birth_intensity'] = 1e-6
tracker_params['init_prob'] = tracker_params['P_D']*tracker_params['birth_intensity']/(tracker_params['clutter_density']+tracker_params['P_D']*tracker_params['birth_intensity'])
tracker_params['conf_threshold'] = 0.99
tracker_params['term_threshold'] = 0.1
tracker_params['visibility_transition_matrix'] = np.array([[0.9, 0.1],[0.52, 0.48]])


measurement_params = dict()
measurement_params['measurement_mapping'] = np.array([[1, 0, 0, 0, 0],[0, 0, 1, 0, 0]])
measurement_params['cart_cov'] = 4.0**2*np.eye(2)
measurement_params['range_cov'] =  2.0**2
measurement_params['bearing_cov'] = ((np.pi/180)*1)**2 


process_params = dict()
process_params['init_mode_probs']=np.array([0.8,0.1,0.1])
process_params['cov_CV_low'] = 0.1**2
process_params['cov_CV_high'] = 1.5**2
process_params['cov_CT'] = 0.02**2
process_params['cov_CV_single'] = 0.5**2

transition_probabilities = [0.99, 0.99, 0.99]

# create transition probability matrix (pi-matrix)
transition_probability_matrix = np.zeros((len(transition_probabilities),len(transition_probabilities)))
for a, transition_probability in enumerate(transition_probabilities):
    transition_probability_matrix[:][a] = (1-transition_probabilities[a])/(len(transition_probabilities)-1)
    transition_probability_matrix[a][a] = transition_probabilities[a]
    try:
        assert(1 - 1e-9 < sum(transition_probability_matrix[:][a]) < 1 + 1e-9)
    except:
        print(transition_probability_matrix[:][a])
        exit()
process_params['pi_matrix'] = transition_probability_matrix