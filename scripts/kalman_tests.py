import csv
import numpy as np
import copy

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


def process_lidar_csv_file(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        csv_rows = [row for row in reader]
        print "%s lidar records" % len(csv_rows)

        n_limit_rows = 1000000

        lidar_obss = []
        bbox_obss = [[],[],[]]

        for i, row in enumerate(csv_rows):
            if i > n_limit_rows - 1:
                break

            time_ns = int(row['time'])
            x, y, z, yaw = float(row['x']), float(row['y']), float(row['z']), float(row['yaw'])
            l, w, h = float(row['l']), float(row['w']), float(row['h'])

            obs = LidarObservation(time_ns * 1e-9, x, y, z, yaw)
            lidar_obss.append(obs)

            bbox_obss[0].append(l)
            bbox_obss[1].append(w)
            bbox_obss[2].append(h)

        yaw = [np.rad2deg(o.yaw) for o in lidar_obss]
        print np.std(yaw)
        #plt.figure(figsize=(16,8))
        #plt.plot(yaw)
        #plt.grid(True)

        return lidar_obss, bbox_obss


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
    # shortenings
    # obs = 'observation'

    base_rate = 1. / 25.
    time_start = lidar_obss[0].timestamp
    print 'base rate(s): %f' % base_rate
    print 'start time(s): %f' % time_start

    n_samples_max = (lidar_obss[-1].timestamp - lidar_obss[0].timestamp) / base_rate + 10
    n_samples_max = int(n_samples_max)

    state_means = []
    state_covs = []
    state_timestamps = []

    fus = FusionUKF(4.358 / 2)
    fus.set_max_timejump(1.)
    fus.set_min_radar_radius(30)

    empty_obss = [EmptyObservation(time_start + base_rate * s_i) for s_i in range(n_samples_max)]
    all_obss = copy.deepcopy(lidar_obss)
    all_obss.extend(radar_obss)
    all_obss.extend(empty_obss)
    obss_timestamps = [o.timestamp for o in all_obss]
    timestamp_sort_inds = np.argsort(obss_timestamps)

    timejump_i = np.random.random_integers(0, len(all_obss) - 1)
    time_shift = 0.
    do_timejump = False

    for i, obs_i in enumerate(timestamp_sort_inds):
        obs = all_obss[obs_i]

        if i == timejump_i and do_timejump:
            time_shift = 1.1

        obs.timestamp += time_shift

        fus.filter(obs)

        mean, cov = fus.last_state_mean, fus.last_state_covar

        if mean is not None:
            state_means.append(mean)
            state_covs.append(cov)
            state_timestamps.append(obs.timestamp)


    # --------- PLOTS -----------

    var = 'ax'
    o_i = FusionUKF.state_var_map[var]
    if var in ['x', 'y', 'vx', 'vy']:
        radar_obss_var = [o.__dict__[var] for o in radar_obss]
    radar_obss_t = [o.timestamp for o in radar_obss]
    if var in ['x', 'y', 'z']:
        lidar_obss_var = [o.__dict__[var] for o in lidar_obss]
    lidar_obss_t = [o.timestamp for o in lidar_obss]
    means = [s[o_i] for s in state_means]
    covs_0 = [o[o_i, o_i] for o in state_covs]
    covs_1 = [o[o_i+1, o_i+1] for o in state_covs]
    covs_2 = [o[o_i+2, o_i+2] for o in state_covs]

    state_timestamps = np.array(state_timestamps)
    radar_obss_t = np.array(radar_obss_t)
    lidar_obss_t = np.array(lidar_obss_t)
    big_num = int(state_timestamps[0] * 1e-2) * 1e+2
    state_timestamps -= big_num
    radar_obss_t -= big_num
    lidar_obss_t -= big_num

    fig, ax1 = plt.subplots(figsize=(16, 8))
    legend = []
    if var in ['x', 'y', 'vx', 'vy']:
        ax1.plot(radar_obss_t, radar_obss_var, 'go')
        legend += ['%s_radar' % var]
    if var in ['x', 'y', 'z']:
        ax1.plot(lidar_obss_t, lidar_obss_var, 'ro')
        legend += ['%s_lidar' % var]
    ax1.plot(state_timestamps, means, 'bo', linewidth=2)
    legend += ['%s_filtered' % var]
    plt.legend(legend, loc=2)

    ax2 = ax1.twinx()
    ax2.plot(state_timestamps, covs_0)
    #ax2.plot(state_timestamps, covs_1)
    #ax2.plot(state_timestamps, covs_2)

    plt.legend(['%s cov' % var]) #, '%s\' cov' % var, '%s\'\' cov' % var])
    plt.grid(True)

    fig.tight_layout()
    plt.show()





def analyze_bbox():
    bbox_arr = np.array(bbox_obss).T
    pos_arr = np.zeros((len(lidar_obss), 4), np.float32)
    for i, o in enumerate(lidar_obss):
        pos_arr[i] = (o.x, o.y, o.z, o.yaw)

    print 'orig mean ', bbox_arr.mean(axis=0), bbox_arr.std(axis=0)
    #print bbox_arr.min(axis=0)
    #print bbox_arr.max(axis=0)

    bbox_arr_fil = np.zeros_like(bbox_arr)

    fil = BBOXSizeFilter(0.01)

    for i in range(bbox_arr.shape[0]):
        bbox_arr_fil[i] = fil.update(*bbox_arr[i])

    print 'filt mean', bbox_arr_fil.mean(axis=0), bbox_arr_fil.std(axis=0)

    var = 0
    plt.plot(figsize=(16, 8))
    plt.plot(bbox_arr[:,var])
    plt.plot(bbox_arr_fil[:,var])
    #plt.plot(pos_arr[:,0])
    plt.grid(True)
    plt.show()

bag_no = 7
#odometry_obss = process_odometry_csv_file('../odometry_ford0{}.bag.csv'.format(bag_no))
radar_obss = process_radar_csv_file('../radar_pred_ford0{}.bag.csv'.format(bag_no))
lidar_obss, bbox_obss = process_lidar_csv_file('../lidar_pred_ford0{}.bag.csv'.format(bag_no))

analyze_ukf(radar_obss, lidar_obss)
#analyze_bbox()