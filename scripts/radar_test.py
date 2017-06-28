#!/usr/bin/env python2

import numpy as np

import rospy
from didi_pipeline.msg import RadarTracks
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../python'))

import radar



def process_radar_tracks(msg):
    assert msg._type == 'didi_pipeline/RadarTracks'
    
    num_tracks = len(msg.tracks)
    
    acc=[]
    cloud = np.zeros([num_tracks, 3], dtype=np.float32)
    for i, track in enumerate(msg.tracks):
        rad = -np.deg2rad(track.angle)
        
        x = track.range * np.cos(rad)
        y = track.range * np.sin(rad)
        z = 0.
        
        vx = track.rate * np.cos(rad)
        vy = track.rate * np.sin(rad)
        
        cloud[i] = [x, y, z]
    
        if np.abs(y) < 2:
            acc.append(x)
            acc.append(track.power)
            #print vx*3.7, vy*3.7
            #print x, y, z
            
    print acc

    header = Header()
    header.stamp = msg.header.stamp
    header.frame_id = 'radar'
    cloud_msg = pc2.create_cloud_xyz32(header, cloud)
    cloud_msg.width = 1
    cloud_msg.height = num_tracks
    rospy.Publisher('radar_points', PointCloud2, queue_size=1).publish(cloud_msg)


rospy.init_node('radar_node')

rospy.Subscriber('/radar/tracks',
                         RadarTracks,
                         process_radar_tracks)

rospy.spin()