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
    def from_msg(msg, radar_to_lidar=None):
        assert msg._md5sum == '6a2de2f790cb8bb0e149d45d297462f8'
        
        num_tracks = len(msg.tracks)
        stamp = msg.header.stamp.to_sec()
        
        radar_obss = []
        
        for i, track in enumerate(msg.tracks):
            rad = -np.deg2rad(track.angle)
            
            x = track.range * np.cos(rad)
            y = track.range * np.sin(rad)
            z = 0.
            
            if radar_to_lidar:
                x -= radar_to_lidar[0]
                y -= radar_to_lidar[1]
                z -= radar_to_lidar[2]
            
            vx = track.rate * np.cos(rad)
            vy = track.rate * np.sin(rad)
            
            radar_obss.append(RadarObservation(stamp, x, y, z, vx, vy))
        
        return radar_obss
        
        