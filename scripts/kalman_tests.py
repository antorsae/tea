import csv
import numpy as np
from pykalman import AdditiveUnscentedKalmanFilter

import matplotlib
import matplotlib.pyplot as plt
print matplotlib.matplotlib_fname()


class LidarObservation:
    def __init__(self, timestamp, x=None, y=None, z=None, yaw=None):
        # timestamp is seconds
        # x,y,z is meters
        # yaw is radians
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

    def has_xyz(self):
        return self.x is not None and self.y is not None and self.z is not None

    def time(self):
        # time in seconds
        return self.timestamp

    def __repr__(self):
        return 'time: {}, x: {}, y: {}, z: {}, yaw: {}'.format(self.timestamp, self.x, self.y, self.z, self.yaw)


class FusionUKF:
    state_obs_map = {'x': 0, 'y': 3, 'z': 6}

    def __init__(self):
        n_state_dims = 9
        n_obs_dims = 3

        def transition_covariance():
            eps = 1e-3 # this value should be small (acceleration speed noise)
            return eps * np.eye(n_state_dims)

        def observation_covariance():
            eps = .1 # derive from the lidar predictor accuracy
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
        self.last_obs = None
        self.last_obs_time = None
        self.initialized = False

    def transition_function(self, state):
        return self.transition_matrix.dot(state)

    @staticmethod
    def observation_function(state):
        m = FusionUKF.state_obs_map
        return [state[m['x']], state[m['y']], state[m['z']]]

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

    def to_kf_obs(self, obss):
        if not isinstance(obss, list):
            obss = [obss]

        kf_obss = []
        for obs in obss:
            kf_obss.append([obs.x, obs.y, obs.z])

        return kf_obss

    def filter(self, obs):
        can_update = obs.has_xyz()

        if not self.initialized and not can_update:
            return

        # we need initial estimation to feed it to filter_update()
        if not self.initialized:
            if not self.last_obs:
                # need two observations to get a filtered state
                self.last_obs = obs
            else:
                dt = obs.timestamp - self.last_obs.timestamp
                self.update_transition_matrix(dt)

                kf_obss = self.to_kf_obs([self.last_obs, obs])
                state_means, state_covars = self.kf.filter(kf_obss)

                self.last_state_mean, self.last_state_covar = state_means[-1], state_covars[-1]
                self.last_obs = obs
                self.initialized = True

            return

        dt = obs.timestamp - self.last_obs.timestamp
        self.update_transition_matrix(dt)

        #print '---'
        #print dt, obs
        #print '---'

        self.last_state_mean, self.last_state_covar = self.kf.filter_update(
            self.last_state_mean,
            self.last_state_covar,
            observation=self.to_kf_obs(obs)[0] if obs.has_xyz() else None)

        self.last_obs = obs


def run_synthetic_test():
    fus = FusionUKF()

    n = 50
    states, observations = fus.kf.sample(n, [0, 1, 0.1, 0, 0, 0, 0, 0, 0])
    states_, covars_ = fus.kf.filter(observations)
    print states.shape
    print states_.shape
    for i in range(n):
        print np.diag(covars_[i])[0]
        #print fus.observation_function(states_[i])
        #print observations[i]
        print


def process_csv_file():
    with open('../lidar_pred_ford03.bag.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        csv_rows = [row for row in reader]
        print "%s records" % len(csv_rows)

        n_limit_rows = 200

        lidar_obss = []

        for i, row in enumerate(csv_rows):
            if i > n_limit_rows - 1:
                break

            time_ns = int(row['time'])
            x, y, z, yaw = float(row['x']), float(row['y']), float(row['z']), float(row['yaw'])

            obs = LidarObservation(time_ns * 1e-9, x, y, z, yaw)

            lidar_obss.append(obs)

        return lidar_obss


def analyze_ukf(lidar_obss):
    fus = FusionUKF()

    # shortenings
    # obs = 'observation'

    base_rate = 1. / 25.
    time_start = lidar_obss[0].timestamp
    print 'base rate(s): %f' % base_rate
    print 'start time(s): %f' % time_start

    n_samples_max = 1000

    state_means = []
    state_covs = []
    base_timestamps = []
    #print base_timestamps
    lidar_obss_actual = []

    drop_noisy = False
    noisy_period = (60, 120) # in lidar obs numbers

    next_obs_i = 0
    for s_i in range(n_samples_max):
        now = time_start + base_rate * s_i

        # no lidar obs left?
        if now > lidar_obss[-1].timestamp:
            break

        # if approached the next lidar obs
        if now >= lidar_obss[next_obs_i].timestamp:
            obs = lidar_obss[next_obs_i]

            if drop_noisy and lidar_obss[noisy_period[0]].timestamp < obs.timestamp < lidar_obss[noisy_period[1]].timestamp:
                # for noisy observations, remove all data, save time
                obs = LidarObservation(obs.timestamp)
            else:
                lidar_obss_actual.append(obs)

            next_obs_i += 1
        # no lidar obs for this timestamp => create dummy obs
        else:
            obs = LidarObservation(now)

        fus.filter(obs)

        mean, cov = fus.last_state_mean, fus.last_state_covar

        if mean is not None:
            state_means.append(mean)
            state_covs.append(cov)

        base_timestamps.append(now)

    base_skip = len(base_timestamps) - len(state_means)

    print 'first %i base timestamps without estimation' % base_skip
    print '%i/%i lidar observations used' % (len(lidar_obss_actual), len(lidar_obss))

    # --------- PLOTS -----------

    var = 'x'
    o_i = FusionUKF.state_obs_map[var]
    obss = [o.__dict__[var] for o in lidar_obss_actual[1:]] # the first observation is not filtered
    obss_t = [o.timestamp for o in lidar_obss_actual[1:]]
    means = [o[o_i] for o in state_means]
    covs = [o[o_i, o_i] for o in state_covs]


    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(obss_t, obss, 'ro')
    ax1.plot(base_timestamps[base_skip:], means, 'b')
    plt.legend(['%s observed' % var, '%s filtered' % var], loc=2)

    ax2 = ax1.twinx()
    ax2.plot(base_timestamps[base_skip:], covs, 'black')

    plt.legend(['%s covariance' % var])
    plt.grid(True)

    fig.tight_layout()
    plt.show()


lidar_obss = process_csv_file()

analyze_ukf(lidar_obss)