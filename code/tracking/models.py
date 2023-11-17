import numpy as np
from scipy.linalg import block_diag
from tracking import constructs

class MeasurementModel(object):
    """
    Generic measurement model class.
    """
    def __init__(self, measurement_mapping, dimension=2):
        self.__measurement_mapping__ = measurement_mapping
        self.dimension = dimension

    def get_measurement_mapping(self):
        return self.__measurement_mapping__

    """
    Returns the position from a full state.
    """
    def __get_state_position__(self, mean):
        if len(mean) == self.dimension:
            return mean
        else:
            return self.__measurement_mapping__.dot(mean)

    def predict_measurement(self, state):
        z_hat = self.__measurement_mapping__.dot(state.mean)
        S = self.__measurement_mapping__.dot(state.covariance).dot(self.__measurement_mapping__.T)+self.get_measurement_covariance(state.mean)
        return z_hat, S

    def get_measurement_covariance(self, mean):
        return np.zeros((self.dimension, self.dimension))


class CartesianMeasurementModel(MeasurementModel):
    """
    A Cartesian measurement model, equipped Cartesian noise matrix.
    """
    def __init__(self, measurement_mapping, cartesian_covariance, **kwargs):
        super().__init__(measurement_mapping, **kwargs)
        self.__cartesian_covariance__ = cartesian_covariance

    def get_measurement_covariance(self, mean):
        return self.__cartesian_covariance__

class PolarMeasurementModel(MeasurementModel):
    """
    A polar measurement model, equipped polar noise matrix.
    """
    def __init__(self, measurement_mapping, range_covariance, bearing_covariance, **kwargs):
        super().__init__(measurement_mapping, **kwargs)
        self.__range_covariance__ = range_covariance
        self.__bearing_covariance__ = bearing_covariance

    def set_ownship_position(self, ownship_measurement):
        self.__ownship_position__ = self.__get_state_position__(ownship_measurement.mean)

    def get_ownship_position(self, ownship):
        return self.__ownship_position__

    def get_measurement_covariance(self, mean):
        state_position = self.__get_state_position__(mean)
        current_range = np.linalg.norm(state_position-self.__ownship_position__)
        current_bearing = np.arctan2(state_position[1]-self.__ownship_position__[1], state_position[0]-self.__ownship_position__[0])
        return self.__polar2cartesian_covariance__(current_range, current_bearing)

    """
    Converts the polar noise matrix to a Cartesian noise matrix.
    """
    def __polar2cartesian_covariance__(self, r, theta):
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        A = np.array([[-r*s_theta, c_theta],[r*c_theta, s_theta]])
        polar = np.array([[self.__bearing_covariance__, 0],[0, self.__range_covariance__]])
        return np.dot(A, polar).dot(A.T)


class CombinedMeasurementModel(CartesianMeasurementModel, PolarMeasurementModel):
    """
    A combined measurement model, equipped both Cartesian and polar noise matrices.
    """
    def get_measurement_covariance(self, mean):
        return PolarMeasurementModel.get_measurement_covariance(self, mean) + CartesianMeasurementModel.get_measurement_covariance(self, mean)


class ClutterModel(object):
    """
    Generic clutter model class.
    """
    def __init__(self, clutter_density):
        self.__clutter_density__ = clutter_density

class ConstantClutterModel(ClutterModel):
    """
    Constant clutter model class, for when the clutter is assumed to result from
    a homogenous PPP with uniformly distributed points.
    """
    def __init__(self, clutter_density):
        super().__init__(clutter_density)

    def get_clutter_density(self):
        return self.__clutter_density__

class KinematicModel(object):
    """
    Generic kinematic model class.
    """
    def __init__(self, state_transition_covariance):
        self.__state_transition_covariance__ = state_transition_covariance

    def step(self, state, dt):
        predicted_mean, F = self.__get_state_and_F_matrix__(dt, state)
        Q = self.__get_Q_matrix__(dt, state=state)
        predicted_covariance = F.dot(state.covariance).dot(F.T)+Q
        return predicted_mean, predicted_covariance

    def __repr__(self):
        return("{}({:1.4f})".format(self.__class__.__name__, self.__state_transition_covariance__))


class CVModel(KinematicModel):
    """
    Constant Velocity (CV) model class.
    """
    def __init__(self, state_transition_covariance, dimension=5):
        super().__init__(state_transition_covariance)
        self.__dimension__ = dimension

    def __get_state_and_F_matrix__(self, dt, state):
        F_block = np.array([[1, dt],[0, 1]])
        diagonal = (F_block,)*int(self.__dimension__/2) + (1,)*int(self.__dimension__%2)
        F = block_diag(*diagonal)
        mean = state.mean
        #print(mean[-1])
        # print(len(F.dot(mean)))
        return F.dot(mean), F

    def __get_Q_matrix__(self, dt, **kwargs):
        Q_block = np.array([[dt**4/4, dt**3/2],[dt**3/2, dt**2]])*self.__state_transition_covariance__
        # print(Q_block)
        diagonal = (Q_block,)*int(self.__dimension__/2) + (0,)*int(self.__dimension__%2)
        return block_diag(*diagonal)

class CTModel(KinematicModel):
    """
    Coordinated Turn (CT) model class.
    """
    def __init__(self, linear_state_transition_covariance, angular_state_transition_covariance, dimension=5):
        self.__linear_state_transition_covariance__ = linear_state_transition_covariance
        self.__angular_state_transition_covariance__ = angular_state_transition_covariance
        self.__dimension__ = dimension

    def __repr__(self):
        return("{}({:1.4f},{:1.4f})".format(self.__class__.__name__, self.__linear_state_transition_covariance__, self.__angular_state_transition_covariance__))

    def __get_state_and_F_matrix__(self, dt, state):
        x = state.mean
        cov_a = self.__linear_state_transition_covariance__
        cov_omega = self.__angular_state_transition_covariance__
        omega = x[4]
        sin_t_o = np.sin(dt*omega)
        cos_t_o = np.cos(dt*omega)
        if abs(omega) > 0.0001:
            F5 = np.array([
                [(x[1]*(dt*cos_t_o*omega-sin_t_o) + x[3]*(-dt*sin_t_o*omega+1-cos_t_o))/(omega**2)],
                [-dt*sin_t_o*x[1] -dt*cos_t_o*x[3]],
                [(x[1]*(dt*sin_t_o*omega-1+cos_t_o) + x[3]*(dt*cos_t_o*omega-sin_t_o))/(omega**2)],
                [x[1]*dt*cos_t_o -dt*sin_t_o*x[3]],
                [1]
            ])

            F = np.concatenate((np.array([
                [1, sin_t_o/omega, 0, -(1-cos_t_o)/omega],
                [0, cos_t_o, 0, -sin_t_o],
                [0, (1-cos_t_o)/omega, 1, sin_t_o/omega],
                [0, sin_t_o, 0, cos_t_o],
                [0, 0, 0, 0]
            ]),F5),axis=1)

            f_x = np.array([
                x[0]+x[1]*sin_t_o/omega-x[3]*(1-cos_t_o)/omega,
                x[1]*cos_t_o-x[3]*sin_t_o,
                x[1]*(1-cos_t_o)/omega+x[2]+x[3]*sin_t_o/omega,
                x[1]*sin_t_o+x[3]*cos_t_o,
                x[4]
            ])
        else:
            F = np.array([[1, dt, 0, 0, -dt**2*x[3]/2],[0, 1, 0, 0, -dt*x[3]],[0, 0, 1, dt, dt**2*x[1]/2],\
             [0, 0, 0, 1, -dt*x[1]],[0, 0, 0, 0, 1]])
            f_x = F.dot(x)
        return f_x, F

    def __get_Q_matrix__(self, dt, state):
        x = state.mean
        cov_a = self.__linear_state_transition_covariance__
        cov_omega = self.__angular_state_transition_covariance__
        omega = x[4]
        sin_t_o = np.sin(dt*omega)
        cos_t_o = np.cos(dt*omega)
        Q_block = np.array([[dt**4*cov_a/4, dt**3*cov_a/2],[dt**3*cov_a/2, dt**2*cov_a]])
        diagonal = ((Q_block,)*int(self.__dimension__/2) + (dt**2*cov_omega,)*int(self.__dimension__%2))
        return block_diag(*diagonal)
