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


class YawLinear:
    quart = np.pi / 3

    def __init__(self):
        self.last_yaw = None
        self.yaw_lin = None

    def update(self, yaw):
        if self.last_yaw is None:
            self.last_yaw = yaw
            self.yaw_lin = yaw
            return yaw

        d = yaw - self.last_yaw
        if abs(d) > abs(YawLinear.quart):
            d = yaw + self.last_yaw

        self.last_yaw = yaw
        self.yaw_lin += d

        return YawLinear.to_pi2(self.yaw_lin)

    @staticmethod
    def to_pi2(y):
        pi = np.pi
        y %= pi
        if pi / 2 <= y < pi:
            y -= pi
        elif pi <= y < 1.5 * pi:
            y = 2 * pi - y
        elif 1.5 * pi <= y < 2 * pi:
            y -= 2 * pi
        return y


class FusionUKF:
    n_state_dims = 11
    n_lidar_obs_dims = 3
    n_radar_obs_dims = 4

    state_var_map = {'x': 0, 'vx': 1, 'ax': 2,
                     'y': 3, 'vy': 4, 'ay': 5,
                     'z': 6, 'vz': 7, 'az': 8,
                     'yaw': 9, 'vyaw': 10}

    OK = 0
    UNRELIABLE_OBSERVATION = 1
    NOT_INITED = 2
    RESETTED = 3

    def __init__(self, object_radius):
        self.initial_state_covariance = FusionUKF.create_initial_state_covariance()

        self.radar_observation_function = FusionUKF.create_radar_observation_function()
        self.lidar_observation_covariance = FusionUKF.create_lidar_observation_covariance()
        self.radar_observation_covariance = FusionUKF.create_radar_observation_covariance(object_radius)

        self.object_radius = object_radius

        # how much noisy observations to reject before resetting the filter
        self.reject_max = 2

        # radar position measurements are coarse at close distance
        # discard radar observations closer than this
        self.min_radar_radius = 0.


        self.max_timejump = 1e9

        self.reset()

    def set_min_radar_radius(self, min_radius):
        self.min_radar_radius = min_radius

    def set_max_timejump(self, max_timejump):
        self.max_timejump = max_timejump

    @staticmethod
    def create_initial_state_covariance():
        # converges really fast, so don't tweak too carefully
        eps = 10.
        return eps * np.eye(FusionUKF.n_state_dims)

    @staticmethod
    def create_transition_function(dt):
        dt2 = 0.5*dt**2
        return lambda s: [
            s[0] + s[1] * dt + s[2] * dt2,
            s[1] + s[2] * dt,
            s[2],
            s[3] + s[4] * dt + s[5] * dt2,
            s[4] + s[5] * dt,
            s[5],
            # let's don't track z velocity and acceleration, because it's unreliable
            # s[6] + s[7] * dt + s[8] * dt2,
            s[6],
            s[7] + s[8] * dt,
            s[8],
            s[9],
            s[10]
        ]

    @staticmethod
    def lidar_observation_function(s):
        m = FusionUKF.state_var_map
        return [
            s[m['x']],
            s[m['y']],
            s[m['z']],
            s[m['yaw']]
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
            1e-5,   # z
            1e-2,   # vz
            1e-2,   # az
            1e-1,   # yaw
            1e-3    # vyaw
        ])

    @staticmethod
    def create_lidar_observation_covariance():
        return np.diag([0.1,    # x
                        0.1,    # y
                        0.1,    # z
                        0.001   # yaw
                        ])

    @staticmethod
    def create_radar_observation_covariance(object_radius):
        cov_x, cov_vx, cov_y, cov_vy = FusionUKF.calc_radar_covariances(object_radius)
        print cov_x, cov_vx, cov_y, cov_vy

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

    def obs_as_kf_obs(self, obs):
        if isinstance(obs, RadarObservation):
            return [obs.x, obs.vx, obs.y, obs.vy]
        elif isinstance(obs, LidarObservation):
            return [obs.x, obs.y, obs.z, obs.yaw]
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
                    z,      0., 0.,
                    0.,     0.]
        elif isinstance(obs, LidarObservation):
            return [obs.x,  0., 0.,
                    obs.y,  0., 0.,
                    obs.z,  0., 0.,
                    obs.yaw,0.]
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

        who_bad = ''
        if bad_x:
            who_bad += 'x'
        elif bad_y:
            who_bad += 'y'
        elif bad_z:
            who_bad += 'z'

        return who_bad if who_bad != '' else None

    def check_covar(self):
        if not self.initialized:
            return

        state_deviation = np.sqrt(np.diag(self.last_state_covar))

        mul = 1.
        deviation_threshold = mul * self.object_radius

        mask = state_deviation > deviation_threshold
        m = self.state_var_map
        bad_x = mask[m['x']]
        bad_y = mask[m['y']]
        bad_z = mask[m['z']]

        if bad_x:
            return 'x', state_deviation[m['x']]
        elif bad_y:
            return 'y', state_deviation[m['y']]
        elif bad_z:
            return 'z', state_deviation[m['z']]

        return None

    def reset(self):
        self.kf = AdditiveUnscentedKalmanFilter(n_dim_state=self.n_state_dims)

        self.obs_yaw_lin = YawLinear()

        self.last_state_mean = None
        self.last_state_covar = None
        self.last_obs = None
        self.initialized = False
        self.reject_count = 0

    def filter(self, obs):
        if not self.initialized and isinstance(obs, EmptyObservation):
            return self.NOT_INITED

        if isinstance(obs, RadarObservation) and obs.radius() < self.min_radar_radius:
            #print 'rejecting radar observation because its too close'

            return self.UNRELIABLE_OBSERVATION

        #print obs

        # we need initial estimation to feed it to filter_update()
        if not self.initialized:
            if not self.last_obs:
                # need two observations to get a filtered state
                self.last_obs = obs

                return self.NOT_INITED

            last_state_mean = self.obs_as_state(self.last_obs)
            last_state_covar = self.initial_state_covariance
        else:
            last_state_mean = self.last_state_mean
            last_state_covar = self.last_state_covar

        dt = obs.timestamp - self.last_obs.timestamp

        if np.abs(dt) > self.max_timejump:
            print 'Fusion: {}s time jump detected, allowed is {}s. Resetting.'.format(dt, self.max_timejump)
            self.reset()
            return self.RESETTED

        who_is_bad = self.looks_like_noise(obs)
        if who_is_bad is not None:
            print 'Fusion: rejected noisy observation (bad {}): {}'.format(who_is_bad, obs)

            self.reject_count += 1

            if self.reject_count > self.reject_max:
                print 'Fusion: resetting filter because too much noise'
                self.reset()

                return self.RESETTED

            return self.UNRELIABLE_OBSERVATION

        if isinstance(obs, LidarObservation):
            transition_function = self.create_transition_function(dt)
            transition_covariance = self.create_transition_covariance()
            observation_function = self.lidar_observation_function
            observation_covariance = self.lidar_observation_covariance
        elif isinstance(obs, RadarObservation):
            transition_function = self.create_transition_function(dt)
            transition_covariance = self.create_transition_covariance()
            observation_function = self.radar_observation_function
            observation_covariance = self.radar_observation_covariance
        else: # EmptyObservation
            transition_function = self.create_transition_function(dt)
            transition_covariance = self.create_transition_covariance()
            observation_function = None
            observation_covariance = None

        try:
            self.last_state_mean, self.last_state_covar =\
                self.kf.filter_update(
                    last_state_mean,
                    last_state_covar,
                    self.obs_as_kf_obs(obs),
                    transition_function,
                    transition_covariance,
                    observation_function,
                    observation_covariance)
        except:
            print 'Fusion: ====== WARNING! ====== filter_update() failed!'

        bad_covar = self.check_covar()
        if bad_covar is not None:
            print 'Fusion: ({}) resetting filter because too high deviation in {}={}'.format(obs.timestamp, bad_covar[0], bad_covar[1])

            self.reset()

            return self.RESETTED

        self.last_obs = obs
        self.initialized = True

        return self.OK


class MovingAverage:
    def __init__(self, tau=0.5):
        self.tau = tau
        self.reset()

    def update(self, x_new):
        if self.x is None:
            self.x = x_new
        else:
            self.x = self.tau * x_new + (1. - self.tau) * self.x

        return self.x

    def reset(self):
        self.x = None


class BBOXSizeFilter:
    def __init__(self, tau=0.01):
        self.tau = tau
        self.reset()

    def update(self, l, w, h):
        return self.l.update(l), self.w.update(w), self.h.update(h)

    def reset(self):
        self.l = MovingAverage(self.tau)
        self.w = MovingAverage(self.tau)
        self.h = MovingAverage(self.tau)