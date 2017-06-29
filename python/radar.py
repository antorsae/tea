import numpy as np


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
        
        