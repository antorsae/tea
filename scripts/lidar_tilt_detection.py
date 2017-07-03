import numpy as np
import pcl
import rosbag
from sensor_msgs.msg._PointCloud import PointCloud
import sensor_msgs.point_cloud2 as pc2

POINT_LIMIT = 65536
_cloud = np.empty((POINT_LIMIT, 3), dtype=np.float32)



def pass_through(x, y, z):
    inner = 4
    outer = 30
    return inner < np.abs(x) < outer and inner < np.abs(y) < outer


def msg2cloud(msg):
    i = 0
    for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z")):
        if pass_through(x, y, z):
            _cloud[i] = x, y, z
            i += 1
    return np.copy(_cloud[:i])


def calcPitchAndRoll(model):
    # rotation around Y axis
    pitch = np.arctan2(model[0], model[2])

    # rotation around X axis
    roll = np.arctan2(model[1], model[2])

    return pitch, roll


bagpath = '/data/didi/dataset_3/car/testing/ford06.bag'

with rosbag.Bag(bagpath) as bag:
    models = []
    pitch_arr = []
    roll_arr = []
    for topic, msg, t in bag.read_messages():
        if topic == '/velodyne_points':
            cloud = msg2cloud(msg)

            pcloud = pcl.PointCloud(cloud)
            seg = pcloud.make_segmenter_normals(ksearch=50)
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
            seg.set_normal_distance_weight(0.1)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_max_iterations(100)
            seg.set_distance_threshold(0.03)
            indices, model = seg.segment()

            models.append(model)
            print model

            if len(models) > 1000:
                break

    model = np.array(models).mean(axis=0)
    print 'mean ', model

    pitch, roll = calcPitchAndRoll(model)
    print 'pitch ', pitch, np.rad2deg(pitch)
    print 'roll ', roll, np.rad2deg(roll)

