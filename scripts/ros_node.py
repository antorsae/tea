#!/usr/bin/env python

import rospy
import os
import sys
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA
from didi_pipeline.msg import Andres
from sensor_msgs.msg._PointCloud import PointCloud
from visualization_msgs.msg import Marker
import tf as ros_tf

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
import time

if K._backend == 'tensorflow':
    import tensorflow as tf
    tf_segmenter_graph = None
    tf_localizer_graph = None

segmenter_model = None
localizer_model = None
points_per_ring = None
clip_distance = None
sectors = None
pointnet_points = None
segmenter_threshold = None
POINT_LIMIT = 65536
cloud = np.empty((POINT_LIMIT, 5), dtype=np.float32)

POINTS_THRESHOLD = 10 # miniimum number of points to do regression

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
    #print 'number of points: %i' % msg.width * msg.height

    # PERFORMANCE WARNING START
    # this preparation code is super slow b/c it uses generator, ideally the code should receive 3 arrays:
    # lidar_d lidar_h lidar_i already preprocessed in C++
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
    lidar_h = np.empty((sectors, points_per_ring // sectors, len(rings)), dtype=np.float32)
    lidar_i = np.empty((sectors, points_per_ring // sectors, len(rings)), dtype=np.float32)
    s_start = 0
    for sector in range(sectors):
        s_end = s_start + points_per_ring // sectors
        for ring in range(len(rings)):
            lidar_d[sector, :, ring] = lidar[ring, s_start:s_end, 0]
            lidar_h[sector, :, ring] = lidar[ring, s_start:s_end, 1]
            lidar_i[sector, :, ring] = lidar[ring, s_start:s_end, 2]
        s_start = s_end
    # PERFORMANCE WARNING END

    with tf_segmenter_graph.as_default():
        time_seg_infe_start = time.time()
        class_predictions_by_angle = segmenter_model.predict([lidar_d, lidar_h, lidar_i], batch_size = sectors)
        time_seg_infe_end   = time.time()

    print ' Seg inference: %0.3f ms' % ((time_seg_infe_end - time_seg_infe_start)   * 1000.0)

    class_predictions_by_angle = np.squeeze(class_predictions_by_angle.reshape((-1, points_per_ring, len(rings))), axis=0)
    class_predictions_by_angle_idx = np.argwhere(class_predictions_by_angle >= segmenter_threshold)

    if (class_predictions_by_angle_idx.shape[0] > 0):
        segmented_points = lidar_int[class_predictions_by_angle_idx[:,0] + points_per_ring * class_predictions_by_angle_idx[:,1]]
    else:
        segmented_points = np.empty((0,3))

    detection = 0
    centroid  = np.zeros((3))
    box_size  = np.zeros((3))
    yaw       = np.zeros((1))

    segmented_points_cloud_msg = pc2.create_cloud_xyz32(msg.header, segmented_points[:,:3])

    if segmented_points.shape[0] >= POINTS_THRESHOLD:

        detection = 1

        segmented_points_mean = np.mean(segmented_points[:, :3], axis=0)
        angle = np.arctan2(segmented_points_mean[1], segmented_points_mean[0])
        segmented_points = point_utils.rotZ(segmented_points, angle)

        segmented_and_aligned_points_mean = np.mean(segmented_points[:, :3], axis=0)
        segmented_points[:, :3] -= segmented_and_aligned_points_mean
        segmented_points[:,  3] /= 128.

        distance_to_segmented_and_aligned_points = np.linalg.norm(segmented_and_aligned_points_mean[:2])

        segmented_points_resampled = DidiTracklet.resample_lidar(segmented_points[:,:4], pointnet_points)

        segmented_points_resampled = np.expand_dims(segmented_points_resampled, axis=0)
        distance_to_segmented_and_aligned_points = np.expand_dims(distance_to_segmented_and_aligned_points, axis=0)

        with tf_localizer_graph.as_default():
            time_loc_infe_start = time.time()
            centroid, box_size, yaw = localizer_model.predict_on_batch([segmented_points_resampled, distance_to_segmented_and_aligned_points])
            time_loc_infe_end   = time.time()

        centroid = np.squeeze(centroid, axis=0)
        box_size = np.squeeze(box_size, axis=0)
        yaw      = np.squeeze(yaw     , axis=0)

        print ' Reg inference: %0.3f ms' % ((time_loc_infe_end - time_loc_infe_start) * 1000.0)

        centroid += segmented_and_aligned_points_mean
        centroid  = point_utils.rotZ(centroid, -angle)
        yaw       = point_utils.remove_orientation(yaw + angle)
        print(centroid, box_size, yaw)

    # publish message (resend msg)
    publisher = rospy.Publisher(name='my_topic', 
                    data_class=Andres, 
                    queue_size=1)
    my_msg = Andres()
    my_msg.header = msg.header
    my_msg.detection = detection
    my_msg.cloud = segmented_points_cloud_msg
    my_msg.length = box_size[2]
    my_msg.width  = box_size[1]
    my_msg.height = box_size[0]
    my_msg.cx = centroid[0]
    my_msg.cy = centroid[1]
    my_msg.cz = centroid[2]
    publisher.publish(my_msg)
    
    # publish car prediction data as separate regular ROS messages just for vizualization (dunno how to visualize custom messages in rviz)
    publish_rviz_topics = True
    
    if publish_rviz_topics and detection > 0:
        # point cloud
        seg_pnt_pub = rospy.Publisher(name='segmented_car',
                                      data_class=PointCloud2,
                                      queue_size=1)
        seg_msg = PointCloud2()
        seg_pnt_pub.publish(segmented_points_cloud_msg)
        
        # car centroid frame
        yaw_q = ros_tf.transformations.quaternion_from_euler(0, 0, yaw)
        br = ros_tf.TransformBroadcaster()
        br.sendTransform(tuple(centroid), tuple(yaw_q), rospy.Time.now(), 'car_pred_centroid', 'velodyne')
        
        # give bbox different color, depending on the predicted object class
        if detection == 1: # car
            bbox_color = ColorRGBA(r=1., g=1., b=0., a=0.5)
        else: # ped
            bbox_color = ColorRGBA(r=0., g=0., b=1., a=0.5)
        
        # bounding box
        marker = Marker()
        marker.header.frame_id = "car_pred_centroid"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale.x = box_size[2]
        marker.scale.y = box_size[1]
        marker.scale.z = box_size[0]
        marker.color = bbox_color
        marker.lifetime = rospy.Duration()
        pub = rospy.Publisher("car_pred_bbox", Marker, queue_size=10)
        pub.publish(marker)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts bounding box pose of obstacle in lidar point cloud.')
    parser.add_argument('-sm', '--segmenter-model', required=True, help='path to hdf5 model')
    parser.add_argument('-lm', '--localizer-model', required=True, help='path to hdf5 model')
    parser.add_argument('-cd', '--clip-distance', default=50., type=float, help='Clip distance (needs to be consistent with trained model!)')
    parser.add_argument('-c', '--cpu', action='store_true', help='force CPU inference')
    parser.add_argument('-st', '--segmenter-threshold', default=0.5, type=float, help='Segmenter classification threshold')

    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    clip_distance       = args.clip_distance
    segmenter_threshold = args.segmenter_threshold

    import keras.losses
    keras.losses.angle_loss = angle_loss

    if True:
        # segmenter model
        segmenter_model = load_model(args.segmenter_model, compile=False)
        segmenter_model._make_predict_function() # https://github.com/fchollet/keras/issues/6124
        print("segmenter model")
        segmenter_model.summary()
        points_per_ring = segmenter_model.get_input_shape_at(0)[0][1]
        match = re.search(r'lidarnet-seg-rings_(\d+)_(\d+)-sectors_(\d+)-.*\.hdf5', args.segmenter_model)
        rings = range(int(match.group(1)), int(match.group(2)))
        sectors = int(match.group(3))
        points_per_ring *= sectors
        assert len(rings) == segmenter_model.get_input_shape_at(0)[0][2]
        print('Loaded segmenter model with ' + str(points_per_ring) + ' points per ring and ' + str(len(rings)) +
              ' rings from ' + str(rings[0]) + ' to ' + str(rings[-1]) )


        if K._backend == 'tensorflow':
            tf_segmenter_graph = tf.get_default_graph()
            print(tf_segmenter_graph)

    # localizer model

    if True:

        print("localizer model")
        #'lidarnet-pointnet-rings_10_28-epoch78-val_loss0.0269.hdf5'
        localizer_model = load_model(args.localizer_model, compile=False)
        segmenter_model._make_predict_function() # https://github.com/fchollet/keras/issues/6124
        localizer_model.summary()
        # TODO: check consistency against segmenter model (rings)
        pointnet_points = localizer_model.get_input_shape_at(0)[0][1]
        print('Loaded localizer model with ' + str(pointnet_points) + ' points')

        if K._backend == 'tensorflow':
            tf_localizer_graph = tf.get_default_graph()
            print(tf_localizer_graph)



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