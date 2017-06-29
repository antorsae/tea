#!/usr/bin/env python

import rospy
import rosbag
import os
import sys
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from didi_pipeline.msg import Andres
from didi_pipeline.msg import RadarTracks
from sensor_msgs.msg._PointCloud import PointCloud
from visualization_msgs.msg import Marker
import tf as ros_tf
import pcl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'torbusnet'))
sys.path.append(os.path.join(BASE_DIR, '../python'))

import argparse
import provider_didi
from keras import backend as K
from keras.models import load_model
from diditracklet import *
import point_utils
import re
import time
import threading


# =============== MAGIC NUMBERS ====================== #
CAR_SIZE = [4.358, 1.823, 1.484] # https://en.wikipedia.org/wiki/Ford_Focus_(third_generation)
RADAR_TO_LIDAR = [1.5494 - 3.8, 0., 1.27] # as per mkz.urdf.xacro

# =============== Sensor Fusion ====================== #
from fusion import FusionUKF, EmptyObservation, RadarObservation, LidarObservation

g_fusion = FusionUKF(CAR_SIZE[0] * 0.5)
g_fusion_lock = threading.Lock()


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
deinterpolate = False
segmenter_phased    = False
localizer_points_threshold = None # miniimum number of points to do regression
reject_false_positives = False

CLIP_DIST      = (0., 50.)
CLIP_HEIGHT    = (-3., 1.)

POINT_LIMIT = 65536
cloud = np.empty((POINT_LIMIT, 5), dtype=np.float32)

localizer_points_threshold = None # miniimum number of points to do regression

last_known_position = None
last_known_box_size = None

def angle_loss(y_true, y_pred):
    if K._BACKEND == 'theano':
        import theano
        arctan2 = theano.tensor.arctan2
    elif K._BACKEND == 'tensorflow':
        import tensorflow as tf
        arctan2 = tf.atan2

def handle_velodyne_msg(msg, arg=None):
    global tf_segmenter_graph
    global last_known_position, last_known_box_size
    
    verbose = False

    assert msg._type == 'sensor_msgs/PointCloud2'
    
    # PointCloud2 reference http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html
     
    if verbose: print 'stamp: %s' % msg.header.stamp
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
        clip=CLIP_DIST,
        clip_h=CLIP_HEIGHT,
        return_lidar_interpolated=True)

    points_per_sector = points_per_ring // sectors

    _sectors = 2 * sectors if segmenter_phased else sectors

    lidar_d = np.empty((_sectors, points_per_sector, len(rings)), dtype=np.float32)
    lidar_h = np.empty((_sectors, points_per_sector, len(rings)), dtype=np.float32)
    lidar_i = np.empty((_sectors, points_per_sector, len(rings)), dtype=np.float32)
    s_start = 0
    for sector in range(sectors):
        s_end = s_start + points_per_sector
        for ring in range(len(rings)):
            lidar_d[sector, :, ring] = lidar[ring, s_start:s_end, 0]
            lidar_h[sector, :, ring] = lidar[ring, s_start:s_end, 1]
            lidar_i[sector, :, ring] = lidar[ring, s_start:s_end, 2]
        s_start = s_end

    if segmenter_phased:
        s_start = points_per_sector // 2
        for sector in range(sectors-1):
            _sector = sectors + sector
            s_end = s_start + points_per_sector
            for ring in range(len(rings)):
                lidar_d[_sector, :, ring] = lidar[ring, s_start:s_end, 0]
                lidar_h[_sector, :, ring] = lidar[ring, s_start:s_end, 1]
                lidar_i[_sector, :, ring] = lidar[ring, s_start:s_end, 2]
            s_start = s_end

        for ring in range(len(rings)):
            lidar_d[_sectors-1, :points_per_sector//2, ring] = lidar[ring, :points_per_sector//2,                 0]
            lidar_d[_sectors-1, points_per_sector//2:, ring] = lidar[ring, points_per_ring - points_per_sector//2:, 0]
            lidar_h[_sectors-1, :points_per_sector//2, ring] = lidar[ring, :points_per_sector//2,                 1]
            lidar_h[_sectors-1, points_per_sector//2:, ring] = lidar[ring, points_per_ring - points_per_sector//2:, 1]
            lidar_i[_sectors-1, :points_per_sector//2, ring] = lidar[ring, :points_per_sector//2,                 2]
            lidar_i[_sectors-1, points_per_sector//2:, ring] = lidar[ring, points_per_ring - points_per_sector//2:, 2]

    # PERFORMANCE WARNING END

    with tf_segmenter_graph.as_default():
        time_seg_infe_start = time.time()
        class_predictions_by_angle = segmenter_model.predict([lidar_d, lidar_h, lidar_i], batch_size = _sectors)
        time_seg_infe_end   = time.time()

    if verbose: print ' Seg inference: %0.3f ms' % ((time_seg_infe_end - time_seg_infe_start)   * 1000.0)

    if segmenter_phased:
        _class_predictions_by_angle = class_predictions_by_angle.reshape((-1, points_per_ring, len(rings)))
        class_predictions_by_angle  = np.copy(_class_predictions_by_angle[0 , :])
        class_predictions_by_angle[points_per_sector // 2 : points_per_ring - (points_per_sector//2)] += \
            _class_predictions_by_angle[1 , : points_per_ring - points_per_sector]
        class_predictions_by_angle[0 : points_per_sector // 2 ] += \
            _class_predictions_by_angle[1 , points_per_ring - points_per_sector : points_per_ring - (points_per_sector // 2)]
        class_predictions_by_angle[points_per_ring - (points_per_sector // 2) : ] += \
            _class_predictions_by_angle[1 , points_per_ring - (points_per_sector // 2): ]
        class_predictions_by_angle_idx = np.argwhere(class_predictions_by_angle >= (2 * segmenter_threshold))


    else:
        class_predictions_by_angle = np.squeeze(class_predictions_by_angle.reshape((-1, points_per_ring, len(rings))), axis=0)
        class_predictions_by_angle_idx = np.argwhere(class_predictions_by_angle >= segmenter_threshold)

    if (class_predictions_by_angle_idx.shape[0] > 0):
        if deinterpolate:
            deinterpolated_class_predictions_by_angle_idx = np.empty((0,2))
            lidar_d_interpolated = lidar_d.reshape((-1, points_per_ring, len(rings)))[0]
            for ring in range(len(rings)):
                predictions_idx_in_ring = class_predictions_by_angle_idx[class_predictions_by_angle_idx[:,1] == ring]
                if predictions_idx_in_ring.shape[0] > 1:
                    lidar_d_predictions_in_ring = lidar_d_interpolated[ predictions_idx_in_ring[:,0], ring]
                    lidar_d_predictions_in_ring_unique, lidar_d_predictions_in_ring_unique_idx = np.unique(lidar_d_predictions_in_ring, return_index=True)
                    deinterpolated_class_predictions_by_angle_idx_this_ring = \
                        predictions_idx_in_ring[lidar_d_predictions_in_ring_unique_idx]
                    deinterpolated_class_predictions_by_angle_idx = np.concatenate((
                        deinterpolated_class_predictions_by_angle_idx,
                        deinterpolated_class_predictions_by_angle_idx_this_ring))

            #print(deinterpolated_class_predictions_by_angle_idx.shape)
            #print(class_predictions_by_angle_idx.shape)
            class_predictions_by_angle_idx = deinterpolated_class_predictions_by_angle_idx.astype(int)

        segmented_points = lidar_int[class_predictions_by_angle_idx[:,0] + points_per_ring * class_predictions_by_angle_idx[:,1]]

        if reject_false_positives and last_known_position is not None and segmented_points.shape[0] > 2:
            import hdbscan
            clusterer = hdbscan.HDBSCAN( allow_single_cluster=True)
            cluster_labels = clusterer.fit_predict(segmented_points[:,:3])
            number_of_clusters  = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if verbose: print("Clusters " + str(number_of_clusters) + ' from ' + str(segmented_points.shape[0]) + ' points')
            if number_of_clusters > 1:
                unique_clusters = set(cluster_labels)
                closest_cluster_center_of_mass = np.array([1e10,1e10,1e10])
                points_in_closest_cluster = None

                for cluster in unique_clusters:
                    points_in_cluster = segmented_points[cluster_labels == cluster]
                    points_in_cluster_center_of_mass = np.linalg.norm(points_in_cluster[:,:3])
                    if np.linalg.norm(closest_cluster_center_of_mass - last_known_position) >\
                            np.linalg.norm(points_in_cluster_center_of_mass - last_known_position):
                        closest_cluster_center_of_mass = points_in_cluster_center_of_mass
                        points_in_closest_cluster = points_in_cluster

                segmented_points = points_in_closest_cluster
                if verbose: print(segmented_points)

    else:
        segmented_points = np.empty((0,3))

    detection = 0
    centroid  = np.zeros((3))
    box_size  = np.zeros((3))
    yaw       = np.zeros((1))

    if segmented_points.shape[0] >= localizer_points_threshold:

        detection = 1
        
            # filter outlier points
        if True:
            time_start = time.time()
            cloud_orig = pcl.PointCloud(segmented_points[:,:3].astype(np.float32))
            fil = cloud_orig.make_statistical_outlier_filter()
            fil.set_mean_k(50)
            fil.set_std_dev_mul_thresh(1.0)
            inlier_inds = fil.filter_ind()
        
            segmented_points = segmented_points[inlier_inds,:]
        
            if verbose: print 'point cloud filter: {:.2f}ms'.format(1e3*(time.time() - time_start))
            
            filtered_points_xyz = segmented_points[:,:3]

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

        if verbose: print ' Reg inference: %0.3f ms' % ((time_loc_infe_end - time_loc_infe_start) * 1000.0)

        centroid += segmented_and_aligned_points_mean
        centroid  = point_utils.rotZ(centroid, -angle)
        yaw       = point_utils.remove_orientation(yaw + angle)
        if verbose: print(centroid, box_size, yaw)

        last_known_position = centroid
        last_known_box_size = box_size
        
        # FUSION
        with g_fusion_lock:
            observation = LidarObservation(msg.header.stamp.to_sec(), centroid[0], centroid[1], centroid[2], yaw)
            g_fusion.filter(observation)
            
            # get filter centroid position
            if g_fusion.last_state_mean is not None:
                centroid = g_fusion.lidar_observation_function(g_fusion.last_state_mean)
    
    segmented_points_cloud_msg = pc2.create_cloud_xyz32(msg.header, segmented_points[:,:3])

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
       # br.sendTransform(tuple(centroid), tuple(yaw_q), rospy.Time.now(), 'car_pred_centroid', 'velodyne')
        
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
        
        # filtered point cloud
        fil_points_msg = pc2.create_cloud_xyz32(msg.header, filtered_points_xyz)
        rospy.Publisher(name='segmented_filt_car',
                      data_class=PointCloud2,
                      queue_size=1).publish(fil_points_msg)
                      
                      
    return {'detection': detection, 
            'x': centroid[0], 
            'y': centroid[1],
            'z': centroid[2], 
            'l': box_size[2],
            'w': box_size[1],
            'h': box_size[0],
            'yaw': np.squeeze(yaw)}
    
    
def handle_radar_msg(msg):
    assert msg._md5sum == '6a2de2f790cb8bb0e149d45d297462f8'
    
    publish_rviz_topics = True
    
    with g_fusion_lock:
        # do we have any estimation?
        if g_fusion.last_state_mean is not None:
            centroid = g_fusion.lidar_observation_function(g_fusion.last_state_mean)
    
            observations = RadarObservation.from_msg(msg, RADAR_TO_LIDAR, CAR_SIZE[1] * 0.5)
            
            # find nearest observation to current object position estimation
            distance_threshold = CAR_SIZE[0]
            nearest = None
            nearest_dist = 1e9
            for o in observations:
                dist = [o.x - centroid[0], o.y - centroid[1], o.z - centroid[2]]
                dist = np.sqrt(np.array(dist).dot(dist))
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = o
            
            if nearest is not None:
                g_fusion.filter(nearest)
                
                if publish_rviz_topics:
                    centroid = g_fusion.lidar_observation_function(g_fusion.last_state_mean)
                    
                    br = ros_tf.TransformBroadcaster()
                    br.sendTransform(tuple(centroid), (0,0,0,1), rospy.Time.now(), 'car_pred_centroid', 'velodyne')
                    
                    if last_known_box_size is not None:
                        # bounding box
                        marker = Marker()
                        marker.header.frame_id = "car_pred_centroid"
                        marker.header.stamp = rospy.Time.now()
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD
                        marker.scale.x = last_known_box_size[2]
                        marker.scale.y = last_known_box_size[1]
                        marker.scale.z = last_known_box_size[0]
                        marker.color = ColorRGBA(r=1., g=1., b=0., a=0.5)
                        marker.lifetime = rospy.Duration()
                        pub = rospy.Publisher("car_pred_bbox", Marker, queue_size=10)
                        pub.publish(marker)
                
                    
if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='Predicts bounding box pose of obstacle in lidar point cloud.')
    parser.add_argument('--bag', help='path to ros bag')
    parser.add_argument('-sm', '--segmenter-model', required=True, help='path to hdf5 model')
    parser.add_argument('-lm', '--localizer-model', required=True, help='path to hdf5 model')
    parser.add_argument('-c', '--cpu', action='store_true', help='force CPU inference')
    parser.add_argument('-st', '--segmenter-threshold', default=0.5, type=float, help='Segmenter classification threshold')
    parser.add_argument('-sp', '--segmenter-phased', action='store_true', help='Use phased-segmenter')
    parser.add_argument('-lpt', '--localizer-points-threshold', default=10, type=int, help='Number of segmented points to trigger a detection')
    parser.add_argument('-di', '--deinterpolate', action='store_true', help='Deinterpolate prior to regression')
    parser.add_argument('-rfp', '--reject-false-positives', action='store_true', help='Rejects false positives')

    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    segmenter_threshold        = args.segmenter_threshold
    segmenter_phased           = args.segmenter_phased
    localizer_points_threshold = args.localizer_points_threshold
    deinterpolate              = args.deinterpolate
    reject_false_positives     = args.reject_false_positives

    import keras.losses
    keras.losses.angle_loss = angle_loss

    if True:
        # segmenter model
        segmenter_model = load_model(args.segmenter_model, compile=False)
        segmenter_model._make_predict_function() # https://github.com/fchollet/keras/issues/6124
        print("segmenter model")
        segmenter_model.summary()
        points_per_ring = segmenter_model.get_input_shape_at(0)[0][1]
        match = re.search(r'seg-rings_(\d+)_(\d+)-sectors_(\d+)-.*\.hdf5', args.segmenter_model)
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
            
    
    # need to init ros to publish messages
    node_name = 'ros_node'
    rospy.init_node(node_name)
    

    if args.bag: # BAG MODE
        import csv
        lidar_writer = csv.DictWriter(open('lidar_pred_{}.csv'.format(os.path.basename(args.bag)), 'w'), fieldnames=['time','detection','x','y','z','l','w','h','yaw'])
        lidar_writer.writeheader()
        radar_writer = csv.DictWriter(open('radar_pred_{}.csv'.format(os.path.basename(args.bag)), 'w'), fieldnames=['timestamp', 'x','y','z','vx','vy'])
        radar_writer.writeheader()
        
        
        # play ros bag
        with rosbag.Bag(args.bag) as bag:
            for topic, msg, t in bag.read_messages():
                if topic == '/image_raw': # 24HZ
                    # predict object pose with kalman_lidar|kalman_radar;
                    # add pose to tracklet;
                    pass
                
                elif topic == '/velodyne_points' and msg.data: # 10HZ
                    # predict object state(NN);
                    pred = handle_velodyne_msg(msg)
                    
                    # update kalman_lidar;
                    if pred['detection'] > 0:
                        # update kalman_lidar
                        pred['time'] = t
                        lidar_writer.writerow(pred)
                        
                elif topic == '/radar/tracks': # 20HZ
                    # use last kalman_lidar|kalman_radar estimation to extract radar points of the object; 
                    # update kalman_radar;
                    observations = RadarObservation.from_msg(msg, RADAR_TO_LIDAR, CAR_SIZE[1] * 0.5)
                    
                    # in ford03 the obstacle is always +-2m along Y-axis
                    last = None
                    for o in observations:
                        if np.abs(o.y) < 2.:
                            if last:
                                if last.x > o.x:
                                    last = o
                            else:
                                last = o
                    if last: radar_writer.writerow(last.__dict__) 
                    
        
    else: # NODE MODE
        # subscribe to the 
        topics = [('/velodyne_points', PointCloud2, handle_velodyne_msg), 
                  ('/radar/tracks', RadarTracks, handle_radar_msg)]
        
        for t in topics:
            rospy.Subscriber(t[0],
                             t[1],
                             t[2])
        
        # this will start infinite loop
        rospy.spin()