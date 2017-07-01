import numpy as np
from pykalman import AdditiveUnscentedKalmanFilter


class EmptyObservation:
    def __init__(self, timestamp):
        self.timestamp = timestamp

    def __repr__(self):
        return '(e) time: {}'.format(self.timestamp)


class OdometryObservation:
    def __init__(self, timestamp, vx, vy, vz):
        # timestamp in secnods
        # vx, vy, vz in m/s
        self.timestamp = timestamp
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def __repr__(self):
        return '(o) time: {:.6f}, vx: {}, vy: {}, vz: {}'.format(self.timestamp, self.vx, self.vy, self.vz)


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
        return '(l) time: {:.6f}, x: {}, y: {}, z: {}, yaw: {}'.format(self.timestamp, self.x, self.y, self.z, self.yaw)


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

    def radius(self):
        return np.sqrt(self.x**2 + self.y**2)

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
        return '(r) time: {:.6f}, x: {}, y: {}, z: {}, vx: {}, vy: {}'.format(self.timestamp, self.x, self.y, self.z, self.vx, self.vy)
        

class FusionUKF:
    n_state_dims = 9
    n_lidar_obs_dims = 3
    n_radar_obs_dims = 4

    state_var_map = {'x': 0, 'vx': 1, 'ax': 2,
                     'y': 3, 'vy': 4, 'ay': 5,
                     'z': 6, 'vz': 7, 'az': 8}

    def __init__(self, initial_object_radius):
        self.transition_covariance = FusionUKF.create_transition_covariance()
        self.initial_state_covariance = FusionUKF.create_initial_state_covariance()

        self.lidar_observation_function = FusionUKF.create_lidar_observation_function()
        self.radar_observation_function = FusionUKF.create_radar_observation_function()
        self.lidar_observation_covariance = FusionUKF.create_lidar_observation_covariance()
        self.radar_observation_covariance = FusionUKF.create_radar_observation_covariance(initial_object_radius)

        # how much noisy observations to reject before resetting the filter
        self.reject_max = 2

        # radar position measurements are coarse at close distance
        # discard radar observations closer than this
        self.min_radar_radius = 0.

        self.reset()

    def set_min_radar_radius(self, min_radius):
        self.min_radar_radius = min_radius

    @staticmethod
    def create_initial_state_covariance():
        # converges really fast, so don't tweak too carefully
        eps = 1.
        return eps * np.eye(FusionUKF.n_state_dims)

    @staticmethod
    def create_transition_function(dt):
        dt2 = 0.5*dt**2
        F = np.array(
            [[1, dt, dt2, 0,  0,  0,   0,  0,  0],    # x
             [0,  1, dt,  0,  0,  0,   0,  0,  0],    # x'
             [0,  0, 1,   0,  0,  0,   0,  0,  0],    # x''
             [0,  0, 0,   1, dt, dt2,  0,  0,  0],    # y
             [0,  0, 0,   0,  1,  dt,  0,  0,  0],    # y'
             [0,  0, 0,   0,  0,   1,  0,  0,  0],    # y''
             [0,  0, 0,   0,  0,   0,  1, dt, dt2],   # z
             [0,  0, 0,   0,  0,   0,  0,  1,  dt],   # z'
             [0,  0, 0,   0,  0,   0,  0,  0,  1],    # z''
            ], dtype=np.float32)
        return lambda s: F.dot(s)

    @staticmethod
    def create_lidar_observation_function():
        m = FusionUKF.state_var_map
        return lambda s: [
            s[m['x']],
            s[m['y']],
            s[m['z']]
            ]

    @staticmethod
    def create_radar_observation_function():
        m = FusionUKF.state_var_map
        return lambda s: [
            s[m['x']],
            s[m['vx']],
            s[m['y']],
            s[m['vy']]
        ]

    @staticmethod
    def create_transition_covariance():
        return np.diag([
            1e-2,   # x
            1e-1,   # vx
            1e-0,   # ax
            1e-2,   # x
            1e-2,   # vy
            1e-2,   # ay
            1e-2,   # z
            1e-2,   # vz
            1e-2,   # az
        ])

    @staticmethod
    def create_lidar_observation_covariance():
        return np.diag([0.1, 0.1, 0.1])

    @staticmethod
    def create_radar_observation_covariance(object_radius):
        cov_x, cov_vx, cov_y, cov_vy = FusionUKF.calc_radar_covariances(object_radius)
        print cov_x, cov_vx, cov_y, cov_vy

        return np.diag([cov_x, cov_vx, cov_y, cov_vy])
        #return np.diag([cov_x, cov_y])

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
            #return [obs.x, obs.y]
            return [obs.x, obs.vx, obs.y, obs.vy]
        elif isinstance(obs, LidarObservation):
            return [obs.x, obs.y, obs.z]
        elif isinstance(obs, EmptyObservation):
            return None
        else:
            raise ValueError

    @staticmethod
    def obs_as_state(obs):
        if isinstance(obs, RadarObservation):
            z = -0.8 # radar doesn't measure Z-coord, so we need an initial estimation of Z.
            return [obs.x,  0., 0.,
                    obs.y,  0., 0.,
                    z,      0., 0.]
        elif isinstance(obs, LidarObservation):
            return [obs.x,  0., 0.,
                    obs.y,  0., 0.,
                    obs.z,  0., 0.]
        else:
            raise ValueError

    def looks_like_noise(self, obs):
        if not self.initialized or isinstance(obs, EmptyObservation):
            return

        state_mean  = self.last_state_mean
        state_deviation = np.sqrt(np.diag(self.last_state_covar))

        mul = 10.
        deviation_threshold = mul * state_deviation

        oas = np.array(self.obs_as_state(obs))
        oas_deviation = np.abs(oas - state_mean)

        reject_mask = oas_deviation > deviation_threshold
        m = self.state_var_map
        bad_x = reject_mask[m['x']]
        bad_y = reject_mask[m['y']]
        bad_z = reject_mask[m['z']]

        #print '1', oas_deviation
        #print '2', deviation_threshold

        #return False
        return bad_x or bad_y # or bad_z

    def reset(self):
        self.kf = AdditiveUnscentedKalmanFilter(n_dim_state=self.n_state_dims)

        self.last_state_mean = None
        self.last_state_covar = None
        self.last_obs = None
        self.initialized = False
        self.reject_count = 0

    def filter(self, obs):
        if not self.initialized and isinstance(obs, EmptyObservation):
            return

        if isinstance(obs, RadarObservation) and obs.radius() < self.min_radar_radius:
            #print 'rejecting radar observation because its too close'

            return

        if isinstance(obs, LidarObservation):
            observation_function = self.lidar_observation_function
            observation_covariance = self.lidar_observation_covariance
        elif isinstance(obs, RadarObservation):
            observation_function = self.radar_observation_function
            observation_covariance = self.radar_observation_covariance
        else: # EmptyObservation
            observation_function = None
            observation_covariance = None

        #print obs

        # we need initial estimation to feed it to filter_update()
        if not self.initialized:
            if not self.last_obs:
                # need two observations to get a filtered state
                self.last_obs = obs
            else:
                dt = obs.timestamp - self.last_obs.timestamp

                self.last_state_mean, self.last_state_covar =\
                    self.kf.filter_update(
                        self.obs_as_state(self.last_obs),
                        self.initial_state_covariance,
                        self.obs_as_kf_obs(obs),
                        self.create_transition_function(dt),
                        self.transition_covariance,
                        observation_function,
                        observation_covariance)

                self.last_obs = obs
                self.initialized = True

            return

        dt = obs.timestamp - self.last_obs.timestamp

        if self.looks_like_noise(obs):
            print 'rejected noisy %s observation : %s' % ('lidar' if isinstance(obs, LidarObservation) else 'radar', obs)

            self.reject_count += 1

            if self.reject_count > self.reject_max:
                print 'resetting filter because too much noise'
                self.reset()

            return
            #obs = EmptyObservation(obs.timestamp)

        self.last_state_mean, self.last_state_covar =\
            self.kf.filter_update(
                self.last_state_mean,
                self.last_state_covar,
                self.obs_as_kf_obs(obs),
                self.create_transition_function(dt),
                self.transition_covariance,
                observation_function,
                observation_covariance)

        self.last_obs = obs