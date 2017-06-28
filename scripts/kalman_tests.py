import csv
import numpy as np
from pykalman import AdditiveUnscentedKalmanFilter





class FusionUKF:
    def __init__(self):
        n_state_dims = 9
        n_obs_dims = 3

        def transition_covariance():
            eps = 1e-4 # this value should be small (acceleration speed noise)
            return eps * np.eye(n_state_dims)

        def observation_covariance():
            eps = 1. # derive from the lidar predictor accuracy
            return eps * np.eye(n_obs_dims)

        def initial_state_covariance():
            # converges really fast, so don't tweak too carefully
            pos_cov = 1.
            vel_cov = 10.
            acc_cov = 100.
            return np.diag([pos_cov, vel_cov, acc_cov, pos_cov, vel_cov, acc_cov, pos_cov, vel_cov, acc_cov])

        # observation_covariance, initial_state_mean, initial_state_covariance depends on the source (lidar/radar)
        self.kf = AdditiveUnscentedKalmanFilter(
            self.transition_function, self.observation_function,
            transition_covariance(), observation_covariance(),
            initial_state_covariance=initial_state_covariance()
        )

        self.transition_matrix = np.eye(n_state_dims)
        self.last_state_mean = None
        self.last_state_covar = None
        self.last_observations = []
        self.initialized = False

    def transition_function(self, state):
        return self.transition_matrix.dot(state)

    def observation_function(self, state):
        return [state[0], state[3], state[6]]

    def update_transition_matrix(self, dt):
        dt2 = 0.5*dt**2
        self.transition_matrix = np.array(
            [[1, dt, dt2, 0,  0,  0,  0,  0,  0],  # x
             [0,  1,  dt, 0,  0,  0,  0,  0,  0],  # x'
             [0,  0,   1, 0,  0,  0,  0,  0,  0],  # x''
             [0,  0,   0, 1, dt, dt2, 0,  0,  0],  # y
             [0,  0,   0, 0,  1,  dt, 0,  0,  0],  # y'
             [0,  0,   0, 0,  0,   1, 0,  0,  0],  # y''
             [0,  0,   0, 0,  0,   0, 1, dt, dt2], # z
             [0,  0,   0, 0,  0,   0, 0,  1,  dt], # z'
             [0,  0,   0, 0,  0,   0, 0,  0,   1]  # z''
             ], dtype=np.float32)

    def filter(self, observation, dt):
        self.update_transition_matrix(dt)

        if not self.initialized:
            if len(self.last_observations) < 2:
                # need two observations to a get filtered state
                self.last_observations.append(observation)
            else:
                state_means, state_covars = self.kf.filter(self.last_observations)
                self.last_state_mean, self.last_state_covar = state_means[-1], state_covars[-1]
                self.last_observations = []
                self.initialized = True

            return

        self.last_state_mean, self.last_state_covar = self.kf.filter_update(
            self.last_state_mean, self.last_state_covar, observation)


with open('../lidar_pred.csv') as csvfile:
    fus = FusionUKF()

    # synthetic test
    if False:
        n = 50
        states, observations = fus.kf.sample(n, [0, 1, 0.1, 0, 0, 0, 0, 0, 0])
        states_, covars_ = fus.kf.filter(observations)
        print states.shape
        print states_.shape
        for i in range(n):

            #print np.diag(covars_[i])[0]
            #print fus.observation_function(states_[i])
            #print observations[i]
            print

        exit(0)

    reader = csv.DictReader(csvfile)
    for row in reader:
        time = row['time']
        x, y, z, yaw = float(row['x']), float(row['y']), float(row['z']), float(row['yaw'])

        obs = [x, y, z]
        dt = 0.1 # calculate from timestamp diff
        fus.filter(obs, dt)

        mean, cov = fus.last_state_mean, fus.last_state_covar
        if mean is not None:
            #print np.diag(cov)[8]
            print fus.observation_function(mean)
            print obs
            print