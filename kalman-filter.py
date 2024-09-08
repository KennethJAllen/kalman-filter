# pylint: disable=locally-disabled, invalid-name, too-many-instance-attributes, too-many-arguments

import numpy as np
import matplotlib.pyplot as plt

class System:
    """An abstract system class. State represents the true state of the system."""
    def __init__(self, initial_state: np.ndarray, state_covariance=None):
        self.state = initial_state
        self.state_covariance = state_covariance if state_covariance is not None else np.eye(len(initial_state))
        self.validate_dimenstions()

    def get_state(self):
        """The n x 1 state of the system, x."""
        return self.state

    def get_state_covariance(self):
        """The n x n covariance matrix P of the state."""
        return self.state_covariance

    def state_transition_matrix(self) -> np.ndarray:
        """The n x n state transition matrix A.
        Describes the transition from the previous state to the next state."""
        raise NotImplementedError("Subclasses should implement this method.")

    def process_noise(self) -> np.ndarray:
        """The n x 1 normally distributed process noise w representing noise in the state of the system."""
        raise NotImplementedError("Subclasses should implement this method.")

    def process_noise_cov(self) -> np.ndarray:
        """The n x n covairance uncertainty matrix Q from the environment.
        It is assumed that the process noise w is distributed w ~ N(0,Q)."""
        raise NotImplementedError("Subclasses should implement this method.")

    def transformation_matrix(self) -> np.ndarray:
        """The m x n transformation matrix H.
        Transforms a system from the state space to the measurement space."""
        raise NotImplementedError("Subclasses should implement this method.")

    def measurement_noise(self) -> np.ndarray:
        """The m x 1 normally distributed measurement noise v."""
        raise NotImplementedError("Subclasses should implement this method.")

    def measurement_noise_cov(self) -> np.ndarray:
        """The m x m covariance of the measurement noise R. It is assume that v ~ N(0,R)."""
        raise NotImplementedError("Subclasses should implement this method.")

    def control_vector(self) -> np.ndarray:
        """The p x 1 control vector u .
        Represents influence on the state not described by the state itself."""
        raise NotImplementedError("Subclasses should implement this method.")

    def control_matrix(self) -> np.ndarray:
        """The n x p control matrix B which maps control vector u to state space."""
        raise NotImplementedError("Subclasses should implement this method.")

    def update_true_state(self) -> None:
        """Updates the true state of the system."""
        if self.process_noise() is not None:
            noise = self.process_noise()
        else:
            noise = 0
        if self.control_matrix() is not None and self.control_vector() is not None:
            control_update = self.control_matrix() @ self.control_vector()
        else:
            control_update = 0
        self.state = self.state_transition_matrix() @ self.state + control_update + noise

    def get_measurement(self) -> np.ndarray:
        """Get an artificial measurement for the given state."""
        return self.transformation_matrix() @ self.state + self.measurement_noise()

    def validate_dimenstions(self):
        """Validate the dimensions of the system matrices."""
        # validate state dimensions
        state_dims = np.shape(self.state)
        state_dim = state_dims[0] # n
        if state_dims != (state_dim, 1):
            raise ValueError(f"The system state does not have the correct dimensions: {state_dims}")
        # validate state transition matrix dimensions
        state_transition_matrix_dims = np.shape(self.state_transition_matrix())
        if state_transition_matrix_dims != (state_dim, state_dim):
            raise ValueError(f"The state transition matrix does not have the correct dimensions: {state_transition_matrix_dims}")
        # validate processs noise dimensions
        if self.process_noise() is not None:
            process_noise_dims = np.shape(self.process_noise())
            if process_noise_dims != (state_dim, 1):
                raise ValueError(f"The process noise does not have the correct dimensions: {process_noise_dims}.")
        # validate process noise covariance matrix dimensions
        process_noise_cov_dims = np.shape(self.process_noise_cov())
        if process_noise_cov_dims != state_transition_matrix_dims:
            raise ValueError(f"The process noise covariance matrix does not have the correct dimensions: {process_noise_cov_dims}")
        # validate transformation matrix dimensions
        transformation_matrix_dims = np.shape(self.transformation_matrix())
        measurement_dim = transformation_matrix_dims[0] # m
        if transformation_matrix_dims != (measurement_dim, state_dim):
            raise ValueError(f"The transformation matrix does not have the correct dimensions: {transformation_matrix_dims}")
        # validate measurement dimensions
        measurement_dims = np.shape(self.get_measurement())
        if measurement_dims != (measurement_dim, 1):
            raise ValueError(f"The measurement vector does not have the correct dimensions: {measurement_dims}")
        # validate measurement noise dims
        measurement_noise_dims = np.shape(self.measurement_noise())
        if measurement_noise_dims != measurement_dims:
            raise ValueError(f"The measurement noise vector does not have the correct dimensions: {measurement_noise_dims}")
        # validate measurement noise covariance dimensions
        measurement_noise_cov_dims = np.shape(self.measurement_noise_cov())
        if measurement_noise_cov_dims != (measurement_dim, measurement_dim):
            raise ValueError(f"The measurement noise covariance matrix does not have the correct dimensions: {measurement_noise_cov_dims}")
        # validate control vector dimensions
        if self.control_vector() is not None:
            control_vec_dims = np.shape(self.control_vector())
            control_vec_dim = control_vec_dims[0]
            if control_vec_dims != (control_vec_dim, 1):
                raise ValueError(f"The system control vector does not have the correct dimensions: {control_vec_dims}")
        # validate control matrix dimensions
        if self.control_matrix() is not None:
            control_mat_dims = np.shape(self.control_matrix())
            if control_mat_dims != (state_dim, control_vec_dim):
                raise ValueError(f"The system control matrix does not have the correct dimensions: {control_mat_dims}")
        print("Dimensions validated.")

class RandomConstant(System):
    """A state representing a random constant.
    We will assume there is a high measurement noise and low process noise."""
    def __init__(self, std_measurement = 0.1, std_process = 0.00001):
        self.std_measurement = std_measurement
        self.std_process = std_process
        initial_state = np.array([[np.random.uniform(0,1)]])
        super().__init__(initial_state)

    def state_transition_matrix(self) -> np.ndarray:
        """The n x n state transition matrix A.
        Describes the transition from the previous state to the next state."""
        return np.array([[1]])

    def process_noise(self) -> np.ndarray:
        """The n x 1 normally distributed process noise w representing noise in the state of the system."""
        return np.array([[np.random.normal(0, self.std_process)]])

    def process_noise_cov(self) -> np.ndarray:
        """The n x n covairance uncertainty matrix Q from the environment.
        It is assumed that the process noise w is distributed w ~ N(0,Q)."""
        return np.array([[self.std_process ** 2]])

    def transformation_matrix(self) -> np.ndarray:
        """The m x n transformation matrix H.
        Transforms a system from the state space to the measurement space."""
        return np.array([[1]])

    def measurement_noise(self) -> np.ndarray:
        """The m x 1 normally distributed measurement noise v."""
        return np.array([[np.random.normal(0, self.std_measurement)]])

    def measurement_noise_cov(self) -> np.ndarray:
        """The m x m covariance of the measurement noise R. It is assume that v ~ N(0,R)."""
        return np.array([[self.std_measurement ** 2]])

    def control_vector(self) -> np.ndarray:
        """The p x 1 control vector u .
        Represents influence on the state not described by the state itself."""
        return None

    def control_matrix(self) -> np.ndarray:
        """The n x p control matrix B which maps control vector u to state space."""
        return None

class FallingObject(System):
    """A system representing a falling object.
    Solves ode: x'' = g, where g = -9.81."""
    def __init__(self, initial_position: float = 0,
                 initial_velocity: float = 25,
                 dt: float = 0.1,
                 gravity:float = -9.81,
                 std_acc: float  = 0.25,
                 std_measurement: float = 5):
        self.dt = dt
        self.gravity = gravity
        self.std_acc = std_acc
        self.std_measurement = std_measurement
        initial_state = np.array([[initial_position], [initial_velocity]])
        super().__init__(initial_state)

    def state_transition_matrix(self) -> np.ndarray:
        """The n x n state transition matrix A.
        Describes the transition from the previous state to the next state."""
        return np.array([[1,self.dt],[0,1]])
    
    def process_noise(self) -> np.ndarray:
        """The n x 1 normally distributed process noise w representing noise in the state of the system."""
        std_position = self.std_acc * (self.dt)**2 / 2
        std_velocity = self.std_acc *self.dt
        position_noise = np.random.normal(0, std_position)
        velocity_noise = np.random.normal(0, std_velocity)
        return np.array([[position_noise],[velocity_noise]])

    def process_noise_cov(self) -> np.ndarray:
        """The n x n covairance uncertainty matrix Q from the environment.
        It is assumed that the process noise w is distributed w ~ N(0,Q)."""
        std_position = self.std_acc * (self.dt)**2 / 2
        std_velocity = self.std_acc *self.dt
        return np.array([[std_position**2, std_position*std_velocity],[std_position*std_velocity, std_velocity**2]])

    def transformation_matrix(self) -> np.ndarray:
        """The m x n transformation matrix H.
        Transforms a system from the state space to the measurement space."""
        return np.array([[1, 0]])

    def measurement_noise(self) -> np.ndarray:
        """The m x 1 normally distributed measurement noise v."""
        noise = np.random.normal(0, self.std_measurement)
        return np.array([[noise]])

    def measurement_noise_cov(self) -> np.ndarray:
        """The m x m covariance of the measurement noise R. It is assume that v ~ N(0,R)."""
        return np.array([[self.std_measurement**2]])

    def control_vector(self) -> np.ndarray:
        """The p x 1 control vector u.
        Represents influence on the state not described by the state itself."""
        return np.array([[self.gravity]])

    def control_matrix(self) -> np.ndarray:
        """The n x p control matrix B which maps control vector u to state space."""
        return np.array([[self.dt**2 / 2], [self.dt]])

class DampenedOscillator(System):
    """A system representing a dampened oscillator.
    If dampening_coeff**2 < 4*mass*spring_constant, then the system is underdampened.
    If equal, the system is critically dampened. Otherwise it is over dampened.
    Solves ode: mass*x'' + dampening_coeff*x' + spring_constant*x = 0."""
    def __init__(self, initial_position: float = 0,
                 initial_velocity: float = 30,
                 dt: float = 0.1,
                 mass: float = 1,
                 dampening_coeff: float = 2,
                 spring_constant: float = 9,
                 std_acc: float = 0.25,
                 std_measurement: float = 1.2):
        self.dt = dt
        self.mass = mass # m
        self.dampending_coeff = dampening_coeff # b
        self.spring_constant = spring_constant # k
        self.std_acc = std_acc
        self.std_measurement = std_measurement
        initial_state = np.array([[initial_position], [initial_velocity]])
        super().__init__(initial_state)

    def state_transition_matrix(self) -> np.ndarray:
        """The n x n state transition matrix A.
        Describes the transition from the previous state to the next state."""
        m = self.mass
        b = self.dampending_coeff
        k = self.spring_constant
        a11 = 1 - (self.dt**2)*k/(2*m)
        a12 = self.dt - (self.dt ** 2)*b/(2 * m)
        a21 = -self.dt*k/m + (self.dt**2)*b*k/(2*m**2)
        a22 = 1 - self.dt*b/m + (self.dt**2)*(b**2-m*k)/(2*m**2)
        return np.array([[a11, a12],[a21, a22]])

    def process_noise(self) -> np.ndarray:
        """The n x 1 normally distributed process noise w representing noise in the state of the system."""
        std_position = self.std_acc * (self.dt)**2 / 2
        std_velocity = self.std_acc *self.dt
        position_noise = np.random.normal(0, std_position)
        velocity_noise = np.random.normal(0, std_velocity)
        return np.array([[position_noise],[velocity_noise]])

    def process_noise_cov(self) -> np.ndarray:
        """The n x n covairance uncertainty matrix Q from the environment.
        It is assumed that the process noise w is distributed w ~ N(0,Q)."""
        std_position = self.std_acc * (self.dt)**2 / 2
        std_velocity = self.std_acc * self.dt
        return np.array([[std_position**2, std_position*std_velocity],[std_position*std_velocity, std_velocity**2]])

    def transformation_matrix(self) -> np.ndarray:
        """The m x n transformation matrix H.
        Transforms a system from the state space to the measurement space."""
        return np.array([[1, 0]])

    def measurement_noise(self) -> np.ndarray:
        """The m x 1 normally distributed measurement noise v."""
        noise = np.random.normal(0, self.std_measurement)
        return np.array([[noise]])

    def measurement_noise_cov(self) -> np.ndarray:
        """The m x m covariance of the measurement noise R. It is assume that v ~ N(0,R)."""
        std_measurement_guess = 0.05 # If the guess is too low, the filter will trust the measurement more
        return np.array([[std_measurement_guess**2]])

    def control_vector(self) -> np.ndarray:
        """The p x 1 control vector u.
        Represents influence on the state not described by the state itself."""
        return None
    def control_matrix(self) -> np.ndarray:
        """The n x p control matrix B which maps control vector u to state space."""
        return None

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

def calculate_errors(states_over_time: tuple[list[float]]) -> None:
    """Calculates the mean square errors for the Kalman prediction and the measurements.
    states_over_time consists of predicted_observables, measuremets, and true_observales."""
    predicted_observables = states_over_time[0]
    measurements = states_over_time[1]
    true_observales = states_over_time[2]
    kalman_mse = ((predicted_observables - true_observales)**2).mean()
    measurement_mse = ((measurements - true_observales)**2).mean()
    return kalman_mse, measurement_mse

def kalman_process(system: System, num_iters: int) -> tuple[np.ndarray]:
    """Exectutes the kalman process for given parameters and number of iterations."""
    kalman = KalmanFilter(system)
    # initialize arrays recording states over time
    true_observales = np.zeros(num_iters)
    measurements = np.zeros(num_iters)
    predicted_observables = np.zeros(num_iters)
    # Update the state of the system and forcast with Kalman filter
    for index in range(num_iters):
        system.update_true_state() # update the actual system state to the next time step.
        measurement = system.get_measurement()
        predicted_state = kalman.update(measurement) # get the next Kalman filter prediction.
        predicted_observable = kalman.system.transformation_matrix() @ predicted_state
        true_observable = kalman.system.transformation_matrix() @ system.state
        # record observables. One records the observables in the first position.
        true_observales[index] = true_observable[0,0]
        measurements[index] = measurement[0,0]
        predicted_observables[index] = predicted_observable[0,0]

    return predicted_observables, measurements, true_observales

def plot_predictions(n_iters: int,
                     dt: float,
                     states_over_time: tuple[list[float]],
                     title: str = "System") -> None:
    """Plots the resulting measurements along with the kalman predictions and true states.
    states_over_time consists of predicted_observables, measuremets, and true_observales."""
    initial_time = 0
    end_time = n_iters * dt + initial_time
    time = np.arange(initial_time, end_time, dt)
    kalman_mse, measurement_mse = calculate_errors(states_over_time)

    predicted_observables = states_over_time[0]
    measurements = states_over_time[1]
    true_observales = states_over_time[2]
    fig = plt.figure()
    plt.plot(time, predicted_observables, label=f"Kalman Filter Position Prediction. MSE: {round(kalman_mse,2)}", color='r', linewidth=1.5)
    fig.suptitle(f"Kalman filter for {title}", fontsize=20)
    plt.scatter(time, measurements, label=f"Measured Position. MSE: {round(measurement_mse,3)}", facecolors='none', color='b')
    plt.plot(time, true_observales, label='True Position', color='y', linewidth=1.5)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Position', fontsize=15)
    plt.legend()
    directory = "images/"
    file_name = title.lower().replace(" ", "_")
    plt.savefig(f"{directory}{file_name}.png")

def main() -> None:
    """Accessor for running the module."""
    n_iters = 50
    random_constant = RandomConstant()
    random_constant_states = kalman_process(random_constant, n_iters)
    plot_predictions(n_iters, 1, random_constant_states, title = "Random Constant")

    dt = 0.1
    falling_object = FallingObject(dt = dt)
    falling_object_states = kalman_process(falling_object, n_iters)
    plot_predictions(n_iters, dt, falling_object_states, title = "Falling Object")
    dampened_oscillator = DampenedOscillator(dt = dt)
    harmonic_oscillator_states = kalman_process(dampened_oscillator, n_iters)
    plot_predictions(n_iters, dt, harmonic_oscillator_states, title = "Dampened Oscillator")

if __name__ == "__main__":
    main()
