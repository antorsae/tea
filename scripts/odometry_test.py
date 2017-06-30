#!/usr/bin/env python2

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import tf as ros_tf

last_rear = None
last_front = None
last_front_t = None
last_yaw = None

def get_position_from_odometry(msg):
    assert isinstance(msg, Odometry)
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z])

def get_speed_from_odometry(msg):
    assert isinstance(msg, Odometry)
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
    assert isinstance(msg, Odometry)

    global last_rear, last_front, last_yaw, last_front_t

    if who == 'rear':
        last_rear = get_position_from_odometry(msg)
    elif who == 'front' and last_rear is not None:
        cur_front = get_position_from_odometry(msg)
        last_yaw = get_yaw(cur_front, last_rear)
        
        if last_front is not None:
            dt = msg.header.stamp.to_sec() - last_front_t
            speed = (cur_front - last_front)/dt
            speed = np.dot(rotMatZ(-last_yaw), speed)
            #speed = np.dot(rotMatZ(-last_yaw), get_speed_from_odometry(msg))
            print speed[1]*100
            
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


topics = {
    'rear': '/objects/capture_vehicle/rear/gps/rtkfix',
    'front': '/objects/capture_vehicle/front/gps/rtkfix',
}

rospy.init_node('odometry_test')

for key in topics:
    rospy.Subscriber(
        topics[key],
        Odometry,
        process_msg,
        key)
    
rospy.spin()