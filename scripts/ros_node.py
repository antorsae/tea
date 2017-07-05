#!/usr/bin/env python

import rospy
import rosbag
import os
import sys
import copy
from sensor_msgs.msg import PointCloud2, Image
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
sys.path.append(os.path.join(BASE_DIR, 'torbusnet/didi-competition/tracklets/python'))
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

from generate_tracklet import *


# =============== MAGIC NUMBERS ====================== #
PEDESTRIAN_SIZE = [0.85, 0.85, 1.708]

RADAR_TO_LIDAR = [1.5494 - 3.8, 0., 1.27] # as per mkz.urdf.xacro

FUSION_MIN_RADAR_RADIUS_DEFAULT = 30.
FUSION_MAX_TIMEJUMP_DEFAULT = 1.

# =============== Sensor Fusion ====================== #
from fusion import *

g_fusion_min_radar_radius = FUSION_MIN_RADAR_RADIUS_DEFAULT
g_fusion_max_timejump = FUSION_MAX_TIMEJUMP_DEFAULT

def create_fusion():
    fus = FusionUKF(2.179)
    fus.set_min_radar_radius(g_fusion_min_radar_radius)
    fus.set_max_timejump(g_fusion_max_timejump)
    return fus

g_fusion = create_fusion()
g_fusion_lock = threading.Lock()

g_pitch_correction = 0.
g_roll_correction = 0.
g_yaw_correction = 0.
g_z_correction = 0.


g_bbox_scale_l = 1.
g_bbox_scale_w = 1.
g_bbox_scale_h = 1.


if K._backend == 'tensorflow':
    import tensorflow as tf
    tf_segmenter_graph = None
    tf_localizer_graph = None

segmenter_model = None
localizer_model = None
points_per_ring = None
rings = None
clip_distance = None
sectors = None
pointnet_points = None
segmenter_threshold = None
deinterpolate = False
segmenter_phased    = False
localizer_points_threshold = None # miniimum number of points to do regression
reject_false_positives = False
verbose = False

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

def init_segmenter(args_segmenter_model):
    global segmenter_model, rings, sectors, points_per_ring, is_ped, tf_segmenter_graph
    segmenter_model = load_model(args_segmenter_model, compile=False)
    segmenter_model._make_predict_function() # https://github.com/fchollet/keras/issues/6124
    print("Loading segmenter model " + args_segmenter_model)
    segmenter_model.summary()
    points_per_ring = segmenter_model.get_input_shape_at(0)[0][1]
    match = re.search(r'lidarnet-(car|ped)-.*seg-rings_(\d+)_(\d+)-sectors_(\d+)-.*\.hdf5', args_segmenter_model)
    is_ped = match.group(1) == 'ped'
    rings = range(int(match.group(2)), int(match.group(3)))
    sectors = int(match.group(4))
    points_per_ring *= sectors
    assert len(rings) == segmenter_model.get_input_shape_at(0)[0][2]
    print('Loaded segmenter model with ' + str(points_per_ring) + ' points per ring and ' + str(len(rings)) +
          ' rings from ' + str(rings[0]) + ' to ' + str(rings[-1]) )

    if K._backend == 'tensorflow':
        tf_segmenter_graph = tf.get_default_graph()
        print(tf_segmenter_graph)
    return

def init_localizer(args_localizer_model):
    global localizer_model, pointnet_points, tf_localizer_graph
    print("Loading localizer model " + args_localizer_model)
    localizer_model = load_model(args_localizer_model, compile=False)
    localizer_model._make_predict_function()  # https://github.com/fchollet/keras/issues/6124
    localizer_model.summary()
    # TODO: check consistency against segmenter model (rings)
    pointnet_points = localizer_model.get_input_shape_at(0)[0][1]
    print('Loaded localizer model with ' + str(pointnet_points) + ' points')

    if K._backend == 'tensorflow':
        tf_localizer_graph = tf.get_default_graph()
        print(tf_localizer_graph)
    return
     
     
def rotXMat(a):
    cos = np.cos(a)
    sin = np.sin(a)
    return np.array([
        [1.,   0.,   0.],
        [0.,  cos, -sin],
        [0.,  sin,  cos],
    ])


def rotYMat(a):
    cos = np.cos(a)
    sin = np.sin(a)
    return np.array([
        [ cos,   0.,  sin],
        [ 0.,    1.,   0.],
        [-sin,   0.,  cos],
    ])

def rotZMat(a):
    cos = np.cos(a)
    sin = np.sin(a)
    return np.array([
        [ cos,   -sin,  0.],
        [ sin,    cos,  0.],
        [0.,   0.,  1.],
    ])


def bbox_size_factor(r):
    max_r = 50
    k = 0.3
    t = .8
    s = 0.9
    r = min(r, max_r)
    return s + k*np.exp(t*(r-max_r))


def handle_velodyne_msg(msg, arg=None):
    global tf_segmenter_graph
    global last_known_position, last_known_box_size, last_known_yaw

    if segmenter_model is None:
        init_segmenter(args.segmenter_model)
    if localizer_model is None:
        init_localizer(args.localizer_model)

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

    lidar, lidar_int, angle_at_edge = DidiTracklet.filter_lidar_rings(
        cloud[:points],
        rings, points_per_ring,
        clip=CLIP_DIST,
        clip_h=CLIP_HEIGHT,
        return_lidar_interpolated=True,
        return_angle_at_edges = True)

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

    if K._backend == 'tensorflow':
        with tf_segmenter_graph.as_default():
            time_seg_infe_start = time.time()
            class_predictions_by_angle = segmenter_model.predict([lidar_d, lidar_h, lidar_i], batch_size = _sectors)
            time_seg_infe_end   = time.time()
    else:
        time_seg_infe_start = time.time()
        class_predictions_by_angle = segmenter_model.predict([lidar_d, lidar_h, lidar_i], batch_size=_sectors)
        time_seg_infe_end = time.time()

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
        if segmenter_threshold == 0.:
            # dynamic thresholding
            number_of_segmented_points = 0
            dyn_threshold = 0.7
            while (number_of_segmented_points < 100) and dyn_threshold >= 0.2:
                class_predictions_by_angle_thresholded = (class_predictions_by_angle >= dyn_threshold)
                number_of_segmented_points = np.sum(class_predictions_by_angle_thresholded)
                dyn_threshold -= 0.1

            class_predictions_by_angle_idx = np.argwhere(class_predictions_by_angle_thresholded)
            # print(dyn_threshold + 0.1, number_of_segmented_points)
        else:
            #for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            #    print(prob, np.sum(class_predictions_by_angle >= prob))
            class_predictions_by_angle_idx = np.argwhere(class_predictions_by_angle >= segmenter_threshold)

    filtered_points_xyz = np.empty((0,3))

    if (class_predictions_by_angle_idx.shape[0] > 0):
        # the idea of de-interpolation is to remove artifacts created by same-neighbor interpolation
        # by checking repeated values (which are going to be nearest-neighbor interpolated values with high prob)
        # for code convenience, we'e just taking the unique indexes as returned by np.unique but we
        # could further improve this by calculating the center of mass on the X axis of the prediction
        # vector (with the unique elements only), and take the index closest to the center for each duplicated stride.
        if deinterpolate:
            #((a.shape[0] - 1 - i_l ) + (i_f)) // 2
            deinterpolated_class_predictions_by_angle_idx = np.empty((0,2))
            lidar_d_interpolated = lidar_d.reshape((-1, points_per_ring, len(rings)))[0]
            for ring in range(len(rings)):
                predictions_idx_in_ring = class_predictions_by_angle_idx[class_predictions_by_angle_idx[:,1] == ring]
                if predictions_idx_in_ring.shape[0] > 1:
                    lidar_d_predictions_in_ring = lidar_d_interpolated[ predictions_idx_in_ring[:,0], ring]
                    _, lidar_d_predictions_in_ring_unique_idx_first = np.unique(lidar_d_predictions_in_ring,       return_index=True)
                    _, lidar_d_predictions_in_ring_unique_idx_last  = np.unique(lidar_d_predictions_in_ring[::-1], return_index=True)
                    lidar_d_predictions_in_ring_unique_idx = \
                        (lidar_d_predictions_in_ring.shape[0] - 1 - lidar_d_predictions_in_ring_unique_idx_last + lidar_d_predictions_in_ring_unique_idx_first ) // 2
                    deinterpolated_class_predictions_by_angle_idx_this_ring = \
                        predictions_idx_in_ring[lidar_d_predictions_in_ring_unique_idx]
                    deinterpolated_class_predictions_by_angle_idx = np.concatenate((
                        deinterpolated_class_predictions_by_angle_idx,
                        deinterpolated_class_predictions_by_angle_idx_this_ring))

            class_predictions_by_angle_idx = deinterpolated_class_predictions_by_angle_idx.astype(int)


        segmented_points = lidar_int[class_predictions_by_angle_idx[:,0] + points_per_ring * class_predictions_by_angle_idx[:,1]]

        # remove capture vehicle, helps in ford01.bag (and doesn't hurt)
        segmented_points = DidiTracklet._remove_capture_vehicle(segmented_points)

        # TODO: use PREDICTED position instead of last known for false positive rejection
        if reject_false_positives and last_known_position is not None and segmented_points.shape[0] > 2:
            original_number_of_points = segmented_points.shape[0]
            time_start = time.time()

            rfp_implementation = 2
            if rfp_implementation == 1:

                import hdbscan
                clusterer = hdbscan.HDBSCAN( allow_single_cluster=True, metric='l2', min_cluster_size=50)
                cluster_labels = clusterer.fit_predict(segmented_points[:,:2])
                number_of_clusters  = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                if verbose: print("Clusters " + str(number_of_clusters) + ' from ' + str(segmented_points.shape[0]) + ' points')
                if number_of_clusters > 1:
                    unique_clusters = set(cluster_labels)
                    closest_cluster_center_of_mass = np.array([1e20,1e20])
                    best_cluster = None
                    for cluster in unique_clusters:
                        points_in_cluster = segmented_points[cluster_labels == cluster]
                        points_in_cluster_center_of_mass = np.mean(points_in_cluster[:,:2], axis=0)
                        if verbose: print(cluster, points_in_cluster.shape, points_in_cluster_center_of_mass)

                        if cluster != -1:
                            if np.linalg.norm(closest_cluster_center_of_mass - last_known_position[:2]) >\
                                    np.linalg.norm(points_in_cluster_center_of_mass - last_known_position[:2]):
                                closest_cluster_center_of_mass = points_in_cluster_center_of_mass
                                best_cluster = cluster

                    #if verbose: print("best cluster", best_cluster, 'last_known_position at ', last_known_position)
                    selected_clusters = [best_cluster]
                    for cluster in unique_clusters:
                        if (cluster != -1) and (cluster != best_cluster):
                            points_in_cluster = segmented_points[cluster_labels == cluster]
                            distances_from_last_known_position = np.linalg.norm(points_in_cluster[:,:2] - closest_cluster_center_of_mass, axis=1)
                            if np.all(distances_from_last_known_position < 2.): # TODO ADJUST
                                selected_clusters.append(cluster)

                    filtered_points_xyz = segmented_points[np.in1d(cluster_labels, selected_clusters, invert=True),:3]
                    segmented_points =  segmented_points[np.in1d(cluster_labels, selected_clusters)]
                    if verbose:
                        print('selected_clusters: ' + str(selected_clusters) + ' with points '+ str(segmented_points.shape[0]) + '/' + str(original_number_of_points))
            else:
                within_tolerance_idx = (((segmented_points[:,0] - last_known_position[0])**2 + (segmented_points[:,0] - last_known_position[0])**2)  <= (5 ** 2))
                filtered_points_xyz = segmented_points[~within_tolerance_idx,:3]
                if filtered_points_xyz.shape[0] < original_number_of_points:
                    segmented_points    = segmented_points[within_tolerance_idx]

            if verbose: print 'clustering filter: {:.2f}ms'.format(1e3*(time.time() - time_start))


    else:
        segmented_points = np.empty((0,3))

    detection = 0
    pose  = np.zeros((3))
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
            
            #filtered_points_xyz = segmented_points[:,:3]

        segmented_points_mean = np.mean(segmented_points[:, :3], axis=0)
        angle = np.arctan2(segmented_points_mean[1], segmented_points_mean[0])
        aligned_points = point_utils.rotZ(segmented_points, angle)

        segmented_and_aligned_points_mean = np.mean(aligned_points[:, :3], axis=0)
        aligned_points[:, :3] -= segmented_and_aligned_points_mean

        distance_to_segmented_and_aligned_points = np.linalg.norm(segmented_and_aligned_points_mean[:2])

        aligned_points_resampled = DidiTracklet.resample_lidar(aligned_points[:,:4], pointnet_points)

        aligned_points_resampled = np.expand_dims(aligned_points_resampled, axis=0)
        distance_to_segmented_and_aligned_points = np.expand_dims(distance_to_segmented_and_aligned_points, axis=0)

        if K._backend == 'tensorflow':
            with tf_localizer_graph.as_default():
                time_loc_infe_start = time.time()
                pose, box_size, yaw = localizer_model.predict_on_batch([aligned_points_resampled, distance_to_segmented_and_aligned_points])
                time_loc_infe_end   = time.time()
        else:
            time_loc_infe_start = time.time()
            pose, box_size, yaw = localizer_model.predict_on_batch(
                [aligned_points_resampled, distance_to_segmented_and_aligned_points])
            time_loc_infe_end = time.time()
            
        pose     = np.squeeze(pose, axis=0)
        box_size = np.squeeze(box_size, axis=0)
        yaw      = np.squeeze(yaw     , axis=0)

        if verbose: print ' Loc inference: %0.3f ms' % ((time_loc_infe_end - time_loc_infe_start) * 1000.0)

        pose += segmented_and_aligned_points_mean
        pose  = point_utils.rotZ(pose, -angle)
        yaw   = point_utils.remove_orientation(yaw + angle)

        pose_angle = np.arctan2(pose[1], pose[0])
        angle_diff = angle_at_edge - pose_angle
        if angle_diff < 0.:
            angle_diff += 2 * np.pi

        # ALI => delta_time is the time difference in milliseconds (0-100) from the start of the lidar
        # scan to the time the object was detected, i don' know if the lidar msg is referenced to be
        # beginning of the scan or the end... so basically adjust the lidar observation for  cases which
        # we need to test:
        # observation_time = msg.header.stamp.to_sec() + delta_time
        # observation_time = msg.header.stamp.to_sec() - delta_time
        # observation_time = msg.header.stamp.to_sec() + 100 msecs - delta_time

        delta_time = 0.1 * angle_diff / (2*np.pi)
        if verbose: print(angle_at_edge, pose_angle, angle_diff)
        if verbose: print(pose, box_size, yaw, delta_time)

        if delta_time < 0:
            print(angle_at_edge, pose_angle, angle_diff, delta_time)

        # fix lidar static tilt
        Rx = rotXMat(np.deg2rad(g_roll_correction))
        Ry = rotYMat(np.deg2rad(g_pitch_correction))
        Rz = rotZMat(np.deg2rad(g_yaw_correction))

        pose     = Rz.dot(Ry.dot(Rx.dot([pose[0], pose[1], pose[2]])))
        pose[2] += g_z_correction
    
        # scale bbox size
        box_size[2] = g_bbox_scale_l * box_size[2]
        box_size[1] = g_bbox_scale_w * box_size[1]
        box_size[0] = g_bbox_scale_h * box_size[0]

        last_known_position = pose
        last_known_box_size = box_size

        # FUSION
        with g_fusion_lock:
            observation = LidarObservation(msg.header.stamp.to_sec(), pose[0], pose[1], pose[2], yaw)
            g_fusion.filter(observation)
            
            
    segmented_points_cloud_msg = pc2.create_cloud_xyz32(msg.header, segmented_points[:,:3])
    

    # publish car prediction data as separate regular ROS messages just for vizualization (dunno how to visualize custom messages in rviz)
    publish_rviz_topics = True
    
    if publish_rviz_topics and detection > 0:
        # point cloud
        seg_pnt_pub = rospy.Publisher(name='segmented_obj',
                                      data_class=PointCloud2,
                                      queue_size=1)
        seg_msg = PointCloud2()
        seg_pnt_pub.publish(segmented_points_cloud_msg)
        
        # car pose frame 
        yaw_q = ros_tf.transformations.quaternion_from_euler(0, 0, yaw)
        br = ros_tf.TransformBroadcaster()
        br.sendTransform(tuple(pose), tuple(yaw_q), rospy.Time.now(), 'obj_lidar_centroid', 'velodyne')
        
        # give bbox different color, depending on the predicted object class
        if detection == 1: # car
            bbox_color = ColorRGBA(r=1., g=1., b=0., a=0.5)
        else: # ped
            bbox_color = ColorRGBA(r=0., g=0., b=1., a=0.5)
        
        # bounding box
        marker = Marker()
        marker.header.frame_id = "obj_lidar_centroid"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.scale.x = box_size[2]
        marker.scale.y = box_size[1]
        marker.scale.z = box_size[0]
        marker.color = bbox_color
        marker.lifetime = rospy.Duration()
        pub = rospy.Publisher("obj_lidar_bbox", Marker, queue_size=10)
        pub.publish(marker)
        
        # filtered point cloud
        fil_points_msg = pc2.create_cloud_xyz32(msg.header, filtered_points_xyz)
        rospy.Publisher(name='segmented_filt_obj',
                      data_class=PointCloud2,
                      queue_size=1).publish(fil_points_msg)
                    
                      
    return {'detection': detection, 
            'x': pose[0], 
            'y': pose[1],
            'z': pose[2], 
            'l': box_size[2],
            'w': box_size[1],
            'h': box_size[0],
            'yaw': np.squeeze(yaw)}
    
    
def handle_radar_msg(msg, dont_fuse):
    assert msg._md5sum == '6a2de2f790cb8bb0e149d45d297462f8'
    
    publish_rviz_topics = True
    
    with g_fusion_lock:
        # do we have any estimation?
        if g_fusion.last_state_mean is not None:
            pose = g_fusion.lidar_observation_function(g_fusion.last_state_mean)
    
            observations = RadarObservation.from_msg(msg, RADAR_TO_LIDAR, 0.9115)
            
            # find nearest observation to current object position estimation
            distance_threshold = 4.4
            nearest = None
            nearest_dist = 1e9
            for o in observations:
                dist = [o.x - pose[0], o.y - pose[1], o.z - pose[2]]
                dist = np.sqrt(np.array(dist).dot(dist))
                
                if dist < nearest_dist and dist < distance_threshold:
                    nearest_dist = dist
                    nearest = o
            
            if nearest is not None:
                # FUSION
                if not dont_fuse:
                    g_fusion.filter(nearest)
                
                if publish_rviz_topics:
                    header = Header()
                    header.frame_id = '/velodyne'
                    header.stamp = rospy.Time.now()
                    point = np.array([[nearest.x, nearest.y, nearest.z]], dtype=np.float32)
                    rospy.Publisher(name='obj_radar_nearest',
                      data_class=PointCloud2,
                      queue_size=1).publish(pc2.create_cloud_xyz32(header, point))
                    
                    #pose = g_fusion.lidar_observation_function(g_fusion.last_state_mean)
                    
                    #br = ros_tf.TransformBroadcaster()
                    #br.sendTransform(tuple(pose), (0,0,0,1), rospy.Time.now(), 'car_fuse_centroid', 'velodyne')
                    
#                     if last_known_box_size is not None:
#                         # bounding box
#                         marker = Marker()
#                         marker.header.frame_id = "car_fuse_centroid"
#                         marker.header.stamp = rospy.Time.now()
#                         marker.type = Marker.CUBE
#                         marker.action = Marker.ADD
#                         marker.scale.x = last_known_box_size[2]
#                         marker.scale.y = last_known_box_size[1]
#                         marker.scale.z = last_known_box_size[0]
#                         marker.color = ColorRGBA(r=1., g=1., b=0., a=0.5)
#                         marker.lifetime = rospy.Duration()
#                         pub = rospy.Publisher("car_fuse_bbox", Marker, queue_size=10)
#                         pub.publish(marker)
                        
                        
def handle_image_msg(msg):
    assert msg._type == 'sensor_msgs/Image'
    
    with g_fusion_lock:
        g_fusion.filter(EmptyObservation(msg.header.stamp.to_sec()))
                    
        if g_fusion.last_state_mean is not None:
            pose = g_fusion.lidar_observation_function(g_fusion.last_state_mean)
                        
            yaw_q = ros_tf.transformations.quaternion_from_euler(0, 0, pose[3])
            br = ros_tf.TransformBroadcaster()
            br.sendTransform(tuple(pose[:3]), tuple(yaw_q), rospy.Time.now(), 'obj_fuse_centroid', 'velodyne')
            
            if last_known_box_size is not None:
                # bounding box
                marker = Marker()
                marker.header.frame_id = "obj_fuse_centroid"
                marker.header.stamp = rospy.Time.now()
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.scale.x = last_known_box_size[2]
                marker.scale.y = last_known_box_size[1]
                marker.scale.z = last_known_box_size[0]
                marker.color = ColorRGBA(r=0., g=1., b=0., a=0.5)
                marker.lifetime = rospy.Duration()
                pub = rospy.Publisher("obj_fuse_bbox", Marker, queue_size=10)
                pub.publish(marker)
                
    
if __name__ == '__main__':        
    parser = argparse.ArgumentParser(description='Predicts bounding box pose of obstacle in lidar point cloud.')
    parser.add_argument('-b', '--bag', help='path to ros bag')
    parser.add_argument('-sm', '--segmenter-model', required=True, help='path to hdf5 model')
    parser.add_argument('-lm', '--localizer-model', required=True, help='path to hdf5 model')
    parser.add_argument('-c', '--cpu', action='store_true', help='force CPU inference')
    parser.add_argument('-st', '--segmenter-threshold', default=0.5, type=float, help='Segmenter classification threshold (0. for dynamic)')
    parser.add_argument('-sp', '--segmenter-phased', action='store_true', help='Use phased-segmenter')
    parser.add_argument('-lpt', '--localizer-points-threshold', default=10, type=int, help='Number of segmented points to trigger a detection')
    parser.add_argument('-di', '--deinterpolate', action='store_true', help='Deinterpolate prior to regression')
    parser.add_argument('-rfp', '--reject-false-positives', action='store_true', help='Rejects false positives')
    parser.add_argument('-nrf', '--no-radar-fuse', action='store_true', help='use radar data in fusion or not')
    parser.add_argument('-rrd', '--record-raw-data', action='store_true', help='record raw data to csv files')
    parser.add_argument('-pc', '--pitch-correction', default=0., help='apply constant pitch rotation to predicted pose')
    parser.add_argument('-rc', '--roll-correction', default=0., help='apply constant roll rotation to predicted pose')
    parser.add_argument('-yc', '--yaw-correction', default=0., help='apply constant yaw rotation to predicted pose')
    parser.add_argument('-zc', '--z-correction', default=0., help='apply constant z offset to predicted pose')
    parser.add_argument('-fmrr', '--fusion-min-radar-radius', default=FUSION_MIN_RADAR_RADIUS_DEFAULT, help='fuse radar scans not closer than this value [meters]')
    parser.add_argument('-fmtj', '--fusion-max-timejump', default=FUSION_MAX_TIMEJUMP_DEFAULT, help='reset fusion if msg time diff is greater than this [s]')
    parser.add_argument('-bsl', '--bbox-scale-length', default=1., help='scale car bbox length')
    parser.add_argument('-bsw', '--bbox-scale-width', default=1., help='scale car bbox length')
    parser.add_argument('-bsh', '--bbox-scale-height', default=1., help='scale car bbox length')

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    segmenter_threshold        = args.segmenter_threshold
    segmenter_phased           = args.segmenter_phased
    localizer_points_threshold = args.localizer_points_threshold
    deinterpolate              = args.deinterpolate
    reject_false_positives     = args.reject_false_positives
    verbose                    = args.verbose

    import keras.losses
    keras.losses.angle_loss = angle_loss


    if K._backend == 'tensorflow':
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        set_session(tf.Session(config=config))

    if args.segmenter_model:
        if K._backend == 'tensorflow':
            init_segmenter(args.segmenter_model)

        # segmenter model


    # localizer model
    if args.localizer_model:
        if K._backend == 'tensorflow':
            init_localizer(args.localizer_model)
            
    
    g_roll_correction = float(args.roll_correction)
    g_pitch_correction = float(args.pitch_correction)
    g_yaw_correction = float(args.yaw_correction)
    g_z_correction = float(args.z_correction)
    
    g_bbox_scale_l = float(args.bbox_scale_length)
    g_bbox_scale_w = float(args.bbox_scale_width)
    g_bbox_scale_h = float(args.bbox_scale_height)
    
    g_fusion_min_radar_radius = float(args.fusion_min_radar_radius)
    g_fusion_max_timejump = float(args.fusion_max_timejump)


    # need to init ros to publish messages
    node_name = 'ros_node'
    rospy.init_node(node_name)
    

    if args.bag: # BAG MODE
        record_raw_data = args.record_raw_data
        
        if record_raw_data:
            import csv
            lidar_writer = csv.DictWriter(open('lidar_pred_{}.csv'.format(os.path.basename(args.bag)), 'w'), fieldnames=['time','detection','x','y','z','l','w','h','yaw'])
            lidar_writer.writeheader()
            radar_writer = csv.DictWriter(open('radar_pred_{}.csv'.format(os.path.basename(args.bag)), 'w'), fieldnames=['timestamp', 'x','y','z','vx','vy'])
            radar_writer.writeheader()
        
        fusion = create_fusion()
        
        bbox_filter = BBOXSizeFilter(0.01)
        
        tracklet_collection = TrackletCollection()
        
        # play ros bag
        with rosbag.Bag(args.bag) as bag:
            def create_tracklet():
                return Tracklet(object_type='Pedestrian' if is_ped else 'Car', 
                                l=0, w=0, h=0, 
                                first_frame = -1)
            
            def finalize_tracklet(tracklet):
                car_size = last_bbox
                object_size = PEDESTRIAN_SIZE if is_ped else car_size
                tracklet.l = object_size[0]
                tracklet.w = object_size[1]
                tracklet.h = object_size[2]
            
            image_msg_num = bag.get_message_count(['/image_raw'])
            image_frame_i = 0
            
            last_bbox = None
            
            # create the first tracklet
            tracklet = create_tracklet()
            
            print 'Start processing messages in {}...'.format(args.bag)
            for topic, msg, t in bag.read_messages():
                if topic == '/image_raw': # 24HZ
                    # predict object pose with kalman_lidar|kalman_radar;
                    # add pose to tracklet;
                    ret = fusion.filter(EmptyObservation(t.to_sec()))
                    
                    # if fusion is not inited, it's likely it was resetted
                    if len(tracklet.poses) and ret == fusion.NOT_INITED:
                        finalize_tracklet(tracklet)
                        tracklet_collection.tracklets.append(tracklet)
                        tracklet = create_tracklet()
                    
                    if fusion.last_state_mean is not None:
                        pose = fusion.lidar_observation_function(fusion.last_state_mean)
                        
                        tracklet_pose = {'tx': pose[0],
                                         'ty': pose[1],
                                         'tz': pose[2],
                                         'rx': 0.,
                                         'ry': 0.,
                                         'rz': pose[3]}
                        tracklet.poses.append(tracklet_pose)
                        
                        if tracklet.first_frame < 0:
                            tracklet.first_frame = image_frame_i
                            
                    image_frame_i += 1
                    
                    if image_frame_i % 100 == 0:
                        print 'Processed {}/{} image frames'.format(image_frame_i, image_msg_num)
                
                elif topic == '/velodyne_points' and msg.data: # 10HZ
                    pred = handle_velodyne_msg(msg)
                    
                    if pred['detection'] > 0:
                        lidar_obs = LidarObservation(t.to_sec(), pred['x'], pred['y'], pred['z'], pred['yaw'])
                        
                        fusion.filter(lidar_obs)
                        
                        bbox = bbox_filter.update(pred['l'], pred['w'], pred['h'])
                        
                        if last_bbox is None:
                            last_bbox = bbox
                        
                        if len(tracklet.poses):
                            finalize_tracklet(tracklet)
                            tracklet_collection.tracklets.append(tracklet)
                            tracklet = create_tracklet()
                        
                        if record_raw_data:
                            data_row = copy.deepcopy(pred)
                            data_row['time'] = t
                            #data_row['l'] = bbox[0]
                            #data_row['w'] = bbox[1]
                            #data_row['h'] = bbox[2]
                            lidar_writer.writerow(data_row)
                        
                elif topic == '/radar/tracks': # 20HZ
                    # use last kalman_lidar|kalman_radar estimation to extract radar points of the object; 
                    # update kalman_radar;
                    observations = RadarObservation.from_msg(msg, RADAR_TO_LIDAR, 2.179)
                    
                    # do we have any estimation?
                    if fusion.last_state_mean is not None:
                        pose = fusion.lidar_observation_function(fusion.last_state_mean)
                
                        # find nearest observation to current object position estimation
                        distance_threshold = 4.4
                        nearest = None
                        nearest_dist = 1e9
                        for o in observations:
                            dist = [o.x - pose[0], o.y - pose[1], o.z - pose[2]]
                            dist = np.sqrt(np.array(dist).dot(dist))
                            #print dist
                            
                            if dist < nearest_dist and dist < distance_threshold:
                                nearest_dist = dist
                                nearest = o
                        
                        if nearest is not None:
                            fusion.filter(nearest)
                            
                            if record_raw_data:
                                radar_writer.writerow(nearest.__dict__)

            print 'Done.'
            
            if len(tracklet.poses):
                finalize_tracklet(tracklet)
                tracklet_collection.tracklets.append(tracklet)
            
            bag_name = os.path.basename(args.bag).split('.')[0]
            tracklet_path = os.path.join(BASE_DIR, '../tracklets/{}'.format(bag_name + '.xml'))
            # replaced -- with unicode counterpart unless XML files is non-compliant
            tracklet_collection.write_xml(tracklet_path, comment=(' '.join(sys.argv[1:]).replace('--', '&#x002D;&#x002D;'))
)
        
    else: # NODE MODE
        # subscribe to the 
        topics = [('/velodyne_points', PointCloud2, handle_velodyne_msg),
                  ('/radar/tracks', RadarTracks, handle_radar_msg, args.no_radar_fuse),
                  ('/image_raw', Image, handle_image_msg)]
        
        for t in topics:
            rospy.Subscriber(*t)
        
        # this will start infinite loop
        rospy.spin()