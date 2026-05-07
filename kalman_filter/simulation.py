# pylint: disable=locally-disabled, invalid-name

import matplotlib.pyplot as plt
import numpy as np

from kalman_filter.filter import KalmanFilter
from kalman_filter.systems import (
    DampenedOscillator,
    ProjectileMotion,
    RandomConstant,
    System,
)


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
    plt.figure(figsize=(10,10))
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
    falling_object = ProjectileMotion(dt = dt)
    falling_object_states = kalman_process(falling_object, n_iters)
    plot_predictions(n_iters, dt, falling_object_states, title = "Projectile Motion")
    dampened_oscillator = DampenedOscillator(dt = dt)
    harmonic_oscillator_states = kalman_process(dampened_oscillator, n_iters)
    plot_predictions(n_iters, dt, harmonic_oscillator_states, title = "Dampened Oscillator")


if __name__ == "__main__":
    main()
