# pylint: disable=locally-disabled, invalid-name, too-many-instance-attributes, too-many-arguments

import numpy as np
import matplotlib.pyplot as plt

class FallingObject:
    """The parameters representing a falling object with contstant acceleration."""
    def __init__(self):
        self.time = 0
        self.dt = 0.1
        self.state_transition_matrix = np.array([[1,self.dt],[0,1]])
        self.control_input_matrix = np.array([[0.5*self.dt**2], [self.dt]])
        self.transformation_matrix = np.array([[1, 0]])
        self.process_cov_noise = np.array([[(self.dt**4)/4, (self.dt**3)/2],
                                           [(self.dt**3)/2, self.dt**2]]) * 0.25**2
        self.measurement_noise_cov = np.array([0.25**2])
        self.state = np.array([[0],[0]])
        self.measured_state = np.array([[0]])
        self.initial_cov = np.eye(2)
        self.control_vector = np.array([[-9.8]])

    def update_state(self):
        """Updates the true and measured state to the next time step."""
        noise = np.random.normal(0, 50)
        self.time = self.time + self.dt
        new_state_position = -9.8*(1/2)*self.time**2
        new_state_velocity = -9.8*self.time
        self.state = np.array([[new_state_position],[new_state_velocity]])
        self.measured_state = self.transformation_matrix @ self.state + noise

class KalmanFilter:
    """Implements the Kalman filter for given input data."""
    def __init__(self, system: FallingObject):
        self.system = system
        self.predicted_state = system.state # x
        self.predicted_cov = system.initial_cov # P
        self.apriori_state = None # x-
        self.apriori_cov = None # P-
        self.kalman_gain = None # K

    def update_apriori_state(self) -> None:
        """Given state vector and control vector, returns a priori state vector estimate."""
        A = self.system.state_transition_matrix
        B = self.system.control_input_matrix
        apriori_state =  A @ self.predicted_state + B @ self.system.control_vector
        self.apriori_state = apriori_state

    def update_apriori_cov(self) -> None:
        """Given state vector and control vector, returns a priori covariance estimate."""
        A = self.system.state_transition_matrix
        Q = self.system.process_cov_noise
        apriori_cov = A @ self.predicted_cov @ A.T + Q
        self.apriori_cov = apriori_cov

    def update_kalman_gain(self) -> None:
        """Get the Kalman gain for a given state."""
        R = self.system.measurement_noise_cov
        H = self.system.transformation_matrix
        numerator = self.apriori_cov @ H.T
        denominator = H @ numerator + R
        kalman_gain = np.linalg.lstsq(denominator.T, numerator.T)[0].T
        self.kalman_gain = kalman_gain

    def update_state(self, measurement: np.ndarray) ->  None:
        """Get the a posteriori state estimate."""
        H = self.system.transformation_matrix
        measurement_residual = measurement - H @ self.apriori_state
        state = self.apriori_state + self.kalman_gain @ measurement_residual
        self.predicted_state = state

    def update_cov(self) ->  None:
        """Get the a posteriori state estimate."""
        H = self.system.transformation_matrix
        n = np.shape(self.kalman_gain)[0]
        I = np.eye(n)
        state_cov = (I - self.kalman_gain @ H) @ self.apriori_cov
        self.predicted_cov = state_cov

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Given a measurement, gives the next prediction via Kalman filter."""
        self.update_apriori_state()
        self.update_apriori_cov()
        self.update_kalman_gain()
        self.update_state(measurement)
        self.update_cov()
        return self.predicted_state

def calculate_errors(states_over_time: tuple[list[float]]) -> None:
    """Calculates the mean square errors for the Kalman prediction and the measurements.
    states_over_time consists of predicted_observables, measuremets, and true_observales."""
    predicted_observables = states_over_time[0]
    measurements = states_over_time[1]
    true_observales = states_over_time[2]
    kalman_mse = ((predicted_observables - true_observales)**2).mean()
    measurement_mse = ((measurements - true_observales)**2).mean()
    return kalman_mse, measurement_mse

def plot_predictions(system: FallingObject, n_iters: int, states_over_time: tuple[list[float]]) -> None:
    """Plots the resulting measurements along with the kalman predictions and true states.
    states_over_time consists of predicted_observables, measuremets, and true_observales."""
    initial_time = system.time
    end_time = n_iters * system.dt + initial_time
    time = np.arange(initial_time, end_time, system.dt)
    kalman_mse, measurement_mse = calculate_errors(states_over_time)

    predicted_observables = states_over_time[0]
    measurements = states_over_time[1]
    true_observales = states_over_time[2]
    fig = plt.figure()
    plt.plot(time, predicted_observables, label=f"Kalman Filter Position Prediction. MSE: {round(kalman_mse,2)}", color='r', linewidth=1.5)
    fig.suptitle('Kalman filter for 1-D falling object.', fontsize=20)
    plt.plot(time, measurements, label=f"Measured Position. MSE: {round(measurement_mse,2)}", color='b',linewidth=0.5)
    plt.plot(time, true_observales, label='True Position', color='y', linewidth=1.5)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Position', fontsize=20)
    plt.legend()
    plt.savefig('kalman_graph.png')

def kalman_process(system: FallingObject, n_iters: int) -> None:
    """Exectutes the kalman process for given parameters and number of iterations."""
    kalman = KalmanFilter(system)
    H = kalman.system.transformation_matrix
    true_observales = np.zeros(n_iters)
    measurements = np.zeros(n_iters)
    predicted_observables = np.zeros(n_iters)
    for index in range(n_iters):
        system.update_state() # update the actual system state to the next time step.
        measurement = system.measured_state
        predicted_state = kalman.update(measurement) # get the next Kalman filter prediction.
        predicted_observable = H @ predicted_state
        true_state = system.state
        true_observable = H @ true_state

        true_observales[index] = true_observable[0,0]
        measurements[index] = measurement[0,0]
        predicted_observables[index] = predicted_observable[0,0]

    states_over_time = predicted_observables, measurements, true_observales
    plot_predictions(system, n_iters, states_over_time)

def main() -> None:
    """Accessor for running the module."""
    n_iters = 100
    system = FallingObject()
    kalman_process(system, n_iters)

if __name__ == "__main__":
    main()
