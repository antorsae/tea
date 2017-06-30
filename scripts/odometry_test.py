#!/usr/bin/env python2

import numpy as np

import rospy, rosbag
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import tf as ros_tf

last_rear = None
last_front = None
last_front_t = None
last_yaw = None

def get_position_from_odometry(msg):
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z])

def get_speed_from_odometry(msg):
    l = msg.twist.twist.linear
    return np.array([l.x, l.y, l.z])

def get_yaw(p1, p2):
    if abs(p1[0] - p2[0]) < 1e-2:
        return 0.
    return np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

def rotMatZ(a):
    cos = np.cos(a)
    sin = np.sin(a)
    return np.array([
        [cos, -sin, 0.],
        [sin, cos,  0.],
        [0,    0,   1.]
    ])

def process_msg(msg, who):
    msg._type == 'nav_msgs/Odometry'

    global last_rear, last_front, last_yaw, last_front_t

    if who == 'rear':
        last_rear = get_position_from_odometry(msg)
    elif who == 'front' and last_rear is not None:
        cur_front = get_position_from_odometry(msg)
        last_yaw = get_yaw(cur_front, last_rear)
        
        twist_lin = np.dot(rotMatZ(-last_yaw), get_speed_from_odometry(msg))
        
        if last_front is not None:
            dt = msg.header.stamp.to_sec() - last_front_t
            speed = (cur_front - last_front)/dt
            speed = np.dot(rotMatZ(-last_yaw), speed)
            print '1', speed
            print '2', twist_lin
            print '3', np.sqrt((speed-twist_lin).dot(speed-twist_lin))
            
            odo = Odometry()
            odo.header.frame_id = '/base_link'
            odo.header.stamp = rospy.Time.now()
            speed_yaw = get_yaw([0,0,0], -speed) #[-speed[0],-speed[1]*100,0])
            speed_yaw_q = ros_tf.transformations.quaternion_from_euler(0, 0, speed_yaw)
            odo.pose.pose.orientation = Quaternion(*list(speed_yaw_q))
            #odo.twist.twist.linear = Point(x=speed[0], y=speed[1], z=speed[2])
            odo.twist.covariance = list(np.eye(6,6).reshape(1,-1).squeeze())
            pub = rospy.Publisher('odo_speed', Odometry, queue_size=1).publish(odo)
        
        #print last_yaw
        last_front = cur_front
        last_front_t = msg.header.stamp.to_sec()
        
        return twist_lin


topics = {
    'rear': '/objects/capture_vehicle/rear/gps/rtkfix',
    'front': '/objects/capture_vehicle/front/gps/rtkfix',
}

rospy.init_node('odometry_test')

if True:
    import csv
    import os, sys
    bagpath = '/data/didi/dataset_3/car/testing/ford03.bag'
    writer = csv.DictWriter(open('../odometry_{}.csv'.format(os.path.basename(bagpath)), 'w'), 
                            fieldnames=['time','vx','vy','vz'])
    writer.writeheader()
    
    with rosbag.Bag(bagpath) as bag:
        for topic, msg, t in bag.read_messages():
            if topic == '/objects/capture_vehicle/rear/gps/rtkfix':
                process_msg(msg, 'rear')
            elif topic == '/objects/capture_vehicle/front/gps/rtkfix':
                v = process_msg(msg, 'front')
                writer.writerow({'time': msg.header.stamp.to_sec(), 'vx': v[0], 'vy': v[1], 'vz': v[2]})
        
else:
    for key in topics:
        rospy.Subscriber(
            topics[key],
            Odometry,
            process_msg,
            key)
        
    rospy.spin()