#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2


def handle_velodyne_msg(msg, arg):
    assert isinstance(msg, PointCloud2)
    
    # PointCloud2 reference http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html
     
    print 'stamp: %s' % msg.header.stamp
    print 'number of points: %i' % msg.width * msg.height
    
    # publish message (resend msg)
    publisher = rospy.Publisher(name='my_topic', 
                    data_class=PointCloud2, 
                    queue_size=1)
    publisher.publish(msg)
    

if __name__ == '__main__':
    node_name = 'ros_node'
    
    rospy.init_node(node_name)
    
    # sys.argv layout:
    # [filepath, argument1, argument2, ..., argumentN, nodename, logpath]
    # where argument{1..N} are arguments you passed when called:
    # roslaunch didi_pipeline ros_node argument1:=value argument2:=value ...
    
    # subscribe to the 
    velodyne_topic = '/velodyne_points'
    data_class = PointCloud2
    callback = handle_velodyne_msg
    callback_arg = {}
    rospy.Subscriber(velodyne_topic,
                     data_class,
                     callback,
                     callback_arg)
    
    # this will start infinite loop
    rospy.spin()