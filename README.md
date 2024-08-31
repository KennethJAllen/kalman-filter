# kalman-filter

An implementation of the Kalman filter for predicting the dynamics of a falling object based on noisy observations.

## How to run

1) Clone the repo.

2) `pip install -r requirements.txt`

3) Execute the file `kalman-filter.py`. This will generate a graph displaying the measured position, actual position, and predicted position via Kalman filter over time. Aditionally, the MSE of the measured position and the predicted position will be printed.

## Output

![Kalman filter results.](/kalman_graph.png)
