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

from radar import RadarObservation


RADAR_TO_LIDAR = [1.5494 - 3.8, 0., 1.27]



def process_radar_tracks(msg):
    assert msg._md5sum == '6a2de2f790cb8bb0e149d45d297462f8'
    
    tracks = RadarObservation.from_msg(msg)
    
    num_tracks = len(tracks)
    
    acc=[]
    cloud = np.zeros([num_tracks, 3], dtype=np.float32)
    for i, track in enumerate(tracks):
        cloud[i] = [track.x, track.y, track.z] - np.array(RADAR_TO_LIDAR)
    
        if np.abs(track.y) < 2:
            #acc.append(track.x)
            #acc.append(track.power)
            print track.vx*3.7, track.vy*3.7
            #print x, y, z
            
    #print acc #msg.header.stamp

    header = Header()
    header.stamp = msg.header.stamp
    header.frame_id = 'velodyne'
    cloud_msg = pc2.create_cloud_xyz32(header, cloud)
    cloud_msg.width = 1
    cloud_msg.height = num_tracks
    rospy.Publisher('radar_points', PointCloud2, queue_size=1).publish(cloud_msg)


rospy.init_node('radar_node')

rospy.Subscriber('/radar/tracks',
                         RadarTracks,
                         process_radar_tracks)

rospy.spin()