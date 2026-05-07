# pylint: disable=locally-disabled, invalid-name

import numpy as np

from kalman_filter.systems import System


class KalmanFilter:
    """Implements the Kalman filter for given input data."""
    def __init__(self, system: System):
        self.system = system
        self.predicted_state = system.get_state() # x
        self.predicted_cov = system.get_state_covariance() # P
        self.apriori_state = None # x-
        self.apriori_cov = None # P-
        self.kalman_gain = None # K

    def update_apriori_state(self) -> None:
        """Given state vector and control vector, returns a priori state vector estimate."""
        A = self.system.state_transition_matrix()
        B = self.system.control_matrix()
        if B is not None and self.system.control_vector() is not None:
            apriori_state =  A @ self.predicted_state + B @ self.system.control_vector()
        else:
            apriori_state =  A @ self.predicted_state
        self.apriori_state = apriori_state

    def update_apriori_cov(self) -> None:
        """Given state vector and control vector, returns a priori covariance estimate."""
        A = self.system.state_transition_matrix()
        Q = self.system.process_noise_cov()
        apriori_cov = A @ self.predicted_cov @ A.T + Q
        self.apriori_cov = apriori_cov

    def update_kalman_gain(self) -> None:
        """Get the Kalman gain for a given state."""
        R = self.system.measurement_noise_cov()
        H = self.system.transformation_matrix()
        numerator = self.apriori_cov @ H.T
        denominator = H @ numerator + R
        kalman_gain = np.linalg.lstsq(denominator.T, numerator.T, rcond=None)[0].T
        self.kalman_gain = kalman_gain

    def update_prediction(self, measurement: np.ndarray) ->  None:
        """Get the a posteriori state prediction."""
        H = self.system.transformation_matrix()
        measurement_residual = measurement - H @ self.apriori_state
        state = self.apriori_state + self.kalman_gain @ measurement_residual
        self.predicted_state = state

    def update_cov(self) ->  None:
        """Get the a posteriori state covariance prediction."""
        H = self.system.transformation_matrix()
        n = np.shape(self.kalman_gain)[0]
        I = np.eye(n)
        state_cov = (I - self.kalman_gain @ H) @ self.apriori_cov
        self.predicted_cov = state_cov

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Given a measurement, gives the next prediction via Kalman filter."""
        self.update_apriori_state()
        self.update_apriori_cov()
        self.update_kalman_gain()
        self.update_prediction(measurement)
        self.update_cov()
        return self.predicted_state
