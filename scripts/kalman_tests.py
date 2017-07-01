import csv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
print matplotlib.matplotlib_fname()

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../python'))

from fusion import *


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


def process_odometry_csv_file(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        csv_rows = [row for row in reader]
        print "%s odometry records" % len(csv_rows)

        odo_obs = []

        for i, row in enumerate(csv_rows):
            time = float(row['time'])
            vx, vy, vz = float(row['vx']), float(row['vy']), float(row['vz'])

            obs = OdometryObservation(time, vx, vy, vz)

            odo_obs.append(obs)

        return odo_obs


def process_lidar_csv_file(filename):
    with open(filename) as csvfile:
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


def process_radar_csv_file(filename):
    with open(filename) as csvfile:
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

    radar_proximity_threshold = 10.

    next_lidar_obs_i = 0
    next_radar_obs_i = 0
    next_odometry_obs_i = 0
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

            next_lidar_obs_i = min(len(lidar_obss) - 1, next_lidar_obs_i + 1)

        if now >= radar_obss[next_radar_obs_i].timestamp:
            radar_obs = radar_obss[next_radar_obs_i]

            next_radar_obs_i = min(len(radar_obss) - 1, next_radar_obs_i + 1)

        if now >= odometry_obss[next_odometry_obs_i].timestamp:
            odometry_obs = odometry_obss[next_odometry_obs_i]

            next_odometry_obs_i = min(len(odometry_obss) - 1, next_odometry_obs_i + 1)

        #radar_obs = None
        #lidar_obs = None
        print odometry_obs

        # discard too close radar readings
        if radar_obs is not None and np.sqrt(radar_obs.x**2 + radar_obs.y**2) < radar_proximity_threshold:
            radar_obs = None

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

    var = 'x'
    #o_i = FusionUKF.state_obs_map[var]
    o_i = {'x': 0, 'y': 1, 'z': 2}[var]
    radar_obss_var = [o.__dict__[var] for o in radar_obss]
    radar_obss_t = [o.timestamp for o in radar_obss]
    lidar_obss_var = [o.__dict__[var] for o in lidar_obss_actual]
    lidar_obss_t = [o.timestamp for o in lidar_obss_actual]
    #means = [o[o_i] for s in state_means]
    means = [fus.lidar_observation_function(s)[o_i] for s in state_means]
    #covs = [o[o_i, o_i] for o in state_covs]
    r_covs = [o[0,0] for o in state_covs]
    alpha_covs = [o[3,3] for o in state_covs]

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(radar_obss_t, radar_obss_var, 'go')
    ax1.plot(lidar_obss_t, lidar_obss_var, 'ro')
    ax1.plot(base_timestamps[base_skip:], means, 'b', linewidth=2)
    plt.legend(['%s_radar' % var, '%s_lidar' % var, '%s_filtered' % var], loc=2)

    ax2 = ax1.twinx()
    #ax2.plot(base_timestamps[base_skip:], covs, 'black')
    ax2.plot(base_timestamps[base_skip:], r_covs, 'black')
    ax2.plot(base_timestamps[base_skip:], alpha_covs, 'magenta')

    #plt.legend(['%s covariance' % var])
    plt.legend(['r covariance', 'alpha covariance'])
    plt.grid(True)

    fig.tight_layout()
    plt.show()


bag_no = 3
odometry_obss = process_odometry_csv_file('../odometry_ford0{}.bag.csv'.format(bag_no))
radar_obss = process_radar_csv_file('../radar_pred_ford0{}.bag.csv'.format(bag_no))
lidar_obss = process_lidar_csv_file('../lidar_pred_ford0{}.bag.csv'.format(bag_no))

analyze_ukf(radar_obss, lidar_obss)