import numpy as np
from pykalman import AdditiveUnscentedKalmanFilter


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


class RadarObservation:
    def __init__(self, timestamp, x=None, y=None, z=None, vx=None, vy=None):
        # timestamp in seconds
        # x,y,z in meters
        # vx,vy in m/s
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        
    @staticmethod
    def from_msg(msg, radar_to_lidar=None, radius_shift=0.):
        assert msg._md5sum == '6a2de2f790cb8bb0e149d45d297462f8'
        
        stamp = msg.header.stamp.to_sec()
        
        radar_obss = []
        
        for i, track in enumerate(msg.tracks):
            ang = -np.deg2rad(track.angle)
            r = track.range + radius_shift
            
            x = r * np.cos(ang)
            y = r * np.sin(ang)
            z = 0.
            
            if radar_to_lidar:
                x -= radar_to_lidar[0]
                y -= radar_to_lidar[1]
                z -= radar_to_lidar[2]
            
            vx = track.rate * np.cos(ang)
            vy = track.rate * np.sin(ang)
            
            radar_obss.append(RadarObservation(stamp, x, y, z, vx, vy))
        
        return radar_obss
    
    def __repr__(self):
        return 'time: {:.6f}, x: {}, y: {}, z: {}, vx: {}, vy: {}'.format(self.timestamp, self.x, self.y, self.z, self.vx, self.vy)
        

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