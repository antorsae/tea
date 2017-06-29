import csv
import numpy as np
from pykalman import AdditiveUnscentedKalmanFilter

import matplotlib
import matplotlib.pyplot as plt
print matplotlib.matplotlib_fname()

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../python'))

from radar import RadarObservation


class EmptyObservation:
    def __init__(self, timestamp):
        self.timestamp = timestamp

    def __repr__(self):
        return 'time: {}'.format(self.timestamp)


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
        return 'time: {:.6f}, x: {}, y: {}, z: {}, yaw: {}'.format(self.timestamp, self.x, self.y, self.z, self.yaw)


class FusionUKF:
    n_state_dims = 9
    n_lidar_obs_dims = 3
    n_radar_obs_dims = 4

    # mapping between observed (by both lidar and radar) variables and their respective state variable indices
    state_obs_map = {'x': 0, 'vx': 1, 'y': 3, 'vy': 4, 'z': 6}

    def __init__(self, initial_object_radius):
        self.transition_covariance = FusionUKF.create_transition_covariance()
        self.lidar_initial_state_covariance = FusionUKF.create_lidar_initial_state_covariance()
        self.radar_initial_state_covariance = FusionUKF.create_radar_initial_state_covariance(initial_object_radius)
        self.lidar_observation_function = FusionUKF.create_lidar_observation_function()
        self.radar_observation_function = FusionUKF.create_radar_observation_function()
        self.lidar_observation_covariance = FusionUKF.create_lidar_observation_covariance()
        self.radar_observation_covariance = FusionUKF.create_radar_observation_covariance(initial_object_radius)

        self.kf = AdditiveUnscentedKalmanFilter(n_dim_state=FusionUKF.n_state_dims)

        self.last_state_mean = None
        self.last_state_covar = None
        self.last_obs = None
        self.last_obs_time = None
        self.initialized = False

    @staticmethod
    def create_lidar_initial_state_covariance():
        # converges really fast, so don't tweak too carefully
        pos_cov = 1.
        vel_cov = 10.
        acc_cov = 100.
        return np.diag([pos_cov, vel_cov, acc_cov, pos_cov, vel_cov, acc_cov, pos_cov, vel_cov, acc_cov])

    @staticmethod
    def create_radar_initial_state_covariance(object_radius):
        # converges really fast, so don't tweak too carefully
        cov_x, cov_vx, cov_y, cov_vy = FusionUKF.calc_radar_covariances(object_radius)
        cov_z = 1.
        cov_vz = 10.
        cov_a = 100.

        return np.diag([cov_x, cov_vx, cov_a, cov_y, cov_vy, cov_a, cov_z, cov_vz, cov_a])

    @staticmethod
    def create_transition_covariance():
        eps = 1e-3  # this value should be small (acceleration speed noise)
        return eps * np.eye(FusionUKF.n_state_dims)

    @staticmethod
    def create_transition_function(dt):
        dt2 = 0.5*dt**2
        F = np.array(
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
        return lambda s: F.dot(s)

    @staticmethod
    def create_lidar_observation_function():
        m = FusionUKF.state_obs_map
        return lambda s: [s[m['x']], s[m['y']], s[m['z']]]

    @staticmethod
    def create_radar_observation_function():
        m = FusionUKF.state_obs_map
        return lambda s: [s[m['x']], s[m['vx']], s[m['y']], s[m['vy']]]

    @staticmethod
    def create_lidar_observation_covariance():
        eps = .1 # derive from the lidar predictor accuracy
        return eps * np.eye(FusionUKF.n_lidar_obs_dims)

    @staticmethod
    def create_radar_observation_covariance(object_radius):
        cov_x, cov_vx, cov_y, cov_vy = FusionUKF.calc_radar_covariances(object_radius)

        return np.diag([cov_x, cov_vx, cov_y, cov_vy])

    @staticmethod
    def calc_radar_covariances(object_radius):
        # object_radius = radius of the circumscribing circle of the tracked object
        sigma_xy = object_radius / 3.
        cov_xy = sigma_xy ** 2

        # jitter derived from http://www.araa.asn.au/acra/acra2015/papers/pap167.pdf
        jitter_v = 0.12
        sigma_v = jitter_v / 3.
        cov_v = sigma_v ** 2

        return cov_xy, cov_v, cov_xy, cov_v

    @staticmethod
    def obs_as_kf_obs(obs):
        if isinstance(obs, RadarObservation):
            return [obs.x, obs.vx, obs.y, obs.vy]
        elif isinstance(obs, LidarObservation):
            return [obs.x, obs.y, obs.z]
        else:
            raise ValueError

    @staticmethod
    def obs_as_state(obs):
        if isinstance(obs, RadarObservation):
            return [obs.x,  obs.vx,     0.,     obs.y,    obs.vy,   0.,     0,      0.,     0.]
        elif isinstance(obs, LidarObservation):
            return [obs.x,      0.,     0.,     obs.y,        0.,   0.,  obs.z,      0.,     0.]
        else:
            raise ValueError

    def filter(self, obs):
        empty_obs = isinstance(obs, EmptyObservation)

        if not self.initialized and empty_obs:
            return

        # we need initial estimation to feed it to filter_update()
        if not self.initialized:
            if not self.last_obs:
                # need two observations to get a filtered state
                self.last_obs = obs
            else:
                dt = obs.timestamp - self.last_obs.timestamp
                #self.update_transition_matrix(dt)

                #kf_obss = self.to_kf_obs([self.last_obs, obs])
                #state_means, state_covars = self.kf.filter(kf_obss)

                #self.last_state_mean, self.last_state_covar = state_means[-1], state_covars[-1]

                prior_state = self.obs_as_state(self.last_obs)

                # radar doesn't measure Z-coord, so we need an initial estimation of Z.
                # Z = -0.8 is good.
                if isinstance(self.last_obs, RadarObservation):
                    prior_state[FusionUKF.state_obs_map['z']] = -0.8

                is_radar = isinstance(obs, RadarObservation)

                self.last_state_mean, self.last_state_covar =\
                    self.kf.filter_update(
                        prior_state,
                        self.radar_initial_state_covariance if is_radar else self.lidar_initial_state_covariance,
                        self.obs_as_kf_obs(obs),
                        self.create_transition_function(dt),
                        self.transition_covariance,
                        self.radar_observation_function if is_radar else self.lidar_observation_function,
                        self.radar_observation_covariance if is_radar else self.lidar_observation_covariance)

                self.last_obs = obs
                self.initialized = True

            return

        dt = obs.timestamp - self.last_obs.timestamp
        #self.update_transition_matrix(dt)

        #print '---'
        #print dt, obs
        #print '---'

        #print obs
        # self.last_state_mean, self.last_state_covar = self.kf.filter_update(
        #     self.last_state_mean,
        #     self.last_state_covar,
        #     observation=self.to_kf_obs(obs)[0] if not empty_obs else None)

        is_radar = isinstance(obs, RadarObservation)

        self.last_state_mean, self.last_state_covar =\
            self.kf.filter_update(
                self.last_state_mean,
                self.last_state_covar,
                self.obs_as_kf_obs(obs) if not empty_obs else None,
                self.create_transition_function(dt),
                self.transition_covariance,
                self.radar_observation_function if is_radar else self.lidar_observation_function,
                self.radar_observation_covariance if is_radar else self.lidar_observation_covariance)

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


def process_lidar_csv_file():
    with open('../lidar_pred_ford03.bag.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        csv_rows = [row for row in reader]
        print "%s lidar records" % len(csv_rows)

        n_limit_rows = 1000000

        lidar_obss = []

        for i, row in enumerate(csv_rows):
            if i > n_limit_rows - 1:
                break

            time_ns = int(row['time'])
            x, y, z, yaw = float(row['x']), float(row['y']), float(row['z']), float(row['yaw'])

            obs = LidarObservation(time_ns * 1e-9, x, y, z, yaw)

            lidar_obss.append(obs)

        return lidar_obss


def process_radar_csv_file():
    with open('../radar_pred_ford03.bag.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        csv_rows = [row for row in reader]
        print "%s radar records" % len(csv_rows)

        n_limit_rows = 1000000

        radar_obss = []

        for i, row in enumerate(csv_rows):
            if i > n_limit_rows - 1:
                break

            time = float(row['timestamp'])
            x, y, z, vx, vy = float(row['x']), float(row['y']), float(row['z']), float(row['vx']), float(row['vy'])

            obs = RadarObservation(time, x, y, z, vx, vy)
            #print obs

            radar_obss.append(obs)

        return radar_obss


def analyze_ukf(radar_obss, lidar_obss):
    fus = FusionUKF(4.358/2)

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

    drop_noisy = False
    noisy_period = (60, 120) # in lidar obs numbers
    lidar_obss_actual = []

    next_lidar_obs_i = 0
    next_radar_obs_i = 0
    for s_i in range(n_samples_max):
        now = time_start + base_rate * s_i

        # no lidar obs left?
        if now > lidar_obss[-1].timestamp and now > radar_obss[-1].timestamp:
            break

        lidar_obs = None
        radar_obs = None

        # if approached the next lidar obs
        if now >= lidar_obss[next_lidar_obs_i].timestamp:
            lidar_obs = lidar_obss[next_lidar_obs_i]

            if not drop_noisy or not lidar_obss[noisy_period[0]].timestamp < lidar_obs.timestamp < lidar_obss[noisy_period[1]].timestamp:
                lidar_obss_actual.append(lidar_obs)

            next_lidar_obs_i += 1

        if now >= radar_obss[next_radar_obs_i].timestamp:
            radar_obs = radar_obss[next_radar_obs_i]

            next_radar_obs_i += 1

        #radar_obs = None
        #lidar_obs = None

        if radar_obs:
            fus.filter(radar_obs)
        elif lidar_obs:
            fus.filter(lidar_obs)
        else: # no lidar/radar obs for this timestamp => only predict
            fus.filter(EmptyObservation(now))

        mean, cov = fus.last_state_mean, fus.last_state_covar

        if mean is not None:
            state_means.append(mean)
            state_covs.append(cov)

        base_timestamps.append(now)

    base_skip = len(base_timestamps) - len(state_means)

    print 'first %i base timestamps without estimation' % base_skip
    print '%i/%i lidar observations used' % (len(lidar_obss_actual), len(lidar_obss))

    # --------- PLOTS -----------

    var = 'z'
    o_i = FusionUKF.state_obs_map[var]
    radar_obss_var = [o.__dict__[var] for o in radar_obss]
    radar_obss_t = [o.timestamp for o in radar_obss]
    lidar_obss_var = [o.__dict__[var] for o in lidar_obss_actual]
    lidar_obss_t = [o.timestamp for o in lidar_obss_actual]
    means = [o[o_i] for o in state_means]
    covs = [o[o_i, o_i] for o in state_covs]


    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(radar_obss_t, radar_obss_var, 'go')
    ax1.plot(lidar_obss_t, lidar_obss_var, 'ro')
    ax1.plot(base_timestamps[base_skip:], means, 'b', linewidth=2)
    plt.legend(['%s_radar' % var, '%s_lidar' % var, '%s_filtered' % var], loc=2)

    ax2 = ax1.twinx()
    ax2.plot(base_timestamps[base_skip:], covs, 'black')

    plt.legend(['%s covariance' % var])
    plt.grid(True)

    fig.tight_layout()
    plt.show()


radar_obss = process_radar_csv_file()
lidar_obss = process_lidar_csv_file()

analyze_ukf(radar_obss, lidar_obss)