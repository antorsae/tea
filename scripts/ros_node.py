#!/usr/bin/env python

import rospy
import os
import sys
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from didi_pipeline.msg import Andres

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'torbusnet'))

import argparse
import provider_didi
from keras import backend as K
from keras.models import load_model
from diditracklet import *
import point_utils
import re

if K._backend == 'tensorflow':
    import tensorflow as tf
    tf_segmenter_graph = None

segmenter_model = None
localizer_model = None
points_per_ring = None
clip_distance = None
sectors = None
pointnet_points = None
POINT_LIMIT = 65536
cloud = np.empty((POINT_LIMIT, 5), dtype=np.float32)

def angle_loss(y_true, y_pred):
    if K._BACKEND == 'theano':
        import theano
        arctan2 = theano.tensor.arctan2
    elif K._BACKEND == 'tensorflow': # NOT WORDKING!
        import tensorflow as tf
        arctan2 = tf.atan2

def handle_velodyne_msg(msg, arg):
    global tf_segmenter_graph

    assert isinstance(msg, PointCloud2)
    
    # PointCloud2 reference http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html
     
    print 'stamp: %s' % msg.header.stamp
    print 'number of points: %i' % msg.width * msg.height

    # PERFORMANCE WARNING START
    # this preparation code is super slow b/c it uses generator, ideally the code should receive two arrays:
    # lidar_d and lidar_i already preprocessed in C++
    points = 0
    for x, y, z, intensity, ring in pc2.read_points(msg):
        cloud[points] = x, y, z, intensity, ring
        points += 1

    lidar, lidar_int = DidiTracklet.filter_lidar_rings(
        cloud[:points],
        rings, points_per_ring,
        clip=(0., clip_distance),
        return_lidar_interpolated=True)

    lidar_d = np.empty((sectors, points_per_ring // sectors, len(rings)), dtype=np.float32)
    lidar_i = np.empty((sectors, points_per_ring // sectors, len(rings)), dtype=np.float32)
    s_start = 0
    for sector in range(sectors):
        s_end = s_start + points_per_ring // sectors
        for ring in range(len(rings)):
            lidar_d[sector, :, ring] = lidar[ring, s_start:s_end, 0]
            lidar_i[sector, :, ring] = lidar[ring, s_start:s_end, 2]
        s_start = s_end
    # PERFORMANCE WARNING END

    print(tf_segmenter_graph)
    with tf_segmenter_graph.as_default():
        class_predictions_by_angle = segmenter_model.predict([lidar_d, lidar_i], batch_size = sectors)

    class_predictions_by_angle = class_predictions_by_angle.reshape((-1, points_per_ring, len(rings)))

    print(class_predictions_by_angle.shape)
    # lidar_int is an array that can be accessed (points_per*ring


    # publish message (resend msg)
    publisher = rospy.Publisher(name='my_topic', 
                    data_class=Andres, 
                    queue_size=1)
    my_msg = Andres()
    my_msg.header = msg.header
    my_msg.detection = 0
    my_msg.cloud = msg
    my_msg.length = 1
    my_msg.width = 2.
    my_msg.height = 3.
    my_msg.cx = 4.
    my_msg.cy = 5.
    my_msg.cz = 6.
    publisher.publish(my_msg)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts bounding box pose of obstacle in lidar point cloud.')
    parser.add_argument('-sm', '--segmenter-model', required=True, help='path to hdf5 model')
    parser.add_argument('-lm', '--localizer-model', required=True, help='path to hdf5 model')
    parser.add_argument('-cd', '--clip-distance', default=50., type=float, help='Clip distance (needs to be consistent with trained model!)')
    parser.add_argument('-c', '--cpu', action='store_true', help='force CPU inference')

    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    clip_distance = args.clip_distance

    # segmenter model
    segmenter_model = load_model(args.segmenter_model)
    print("segmenter model")
    segmenter_model.summary()
    points_per_ring = segmenter_model.get_input_shape_at(0)[0][1]
    match = re.search(r'lidarnet-seg-rings_(\d+)_(\d+)-sectors_(\d+)-.*\.hdf5', args.segmenter_model)
    rings = range(int(match.group(1)), int(match.group(2)))
    sectors = int(match.group(3))
    points_per_ring *= sectors
    assert len(rings) == segmenter_model.get_input_shape_at(0)[0][2]

    if K._backend == 'tensorflow':
        tf_segmenter_graph = tf.get_default_graph()
        print(tf_segmenter_graph)

    # localizer model

    if False:
        import keras.losses

        # keras.losses.null_loss = null_loss
        keras.losses.angle_loss = angle_loss

        print("localizer model")
        #'lidarnet-pointnet-rings_10_28-epoch78-val_loss0.0269.hdf5'
        localizer_model = load_model(args.localizer_model)
        localizer_model.summary()
        # TODO: check consistency against segmenter model (rings)
        pointnet_points = localizer_model.get_input_shape_at(0)[0]

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