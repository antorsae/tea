#include "didi_pipeline/typedefs.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <assert.h>


using namespace std;
using namespace pcl;
using namespace didi;


int main(int argv, char **argc)
{
	string filename("/data/didi/dataset_3/car/training/nissan_brief/nissan06.bag");

	pcl::visualization::PCLVisualizer viz;

	rosbag::Bag bag;
	bag.open(filename, rosbag::bagmode::Read);

	vector<string> topics;
	topics.push_back("/velodyne_points");

	rosbag::View view(bag, rosbag::TopicQuery(topics));

	int frames_read{};
	bool once{false};

	for (rosbag::MessageInstance const m : view) {
		if (m.getTopic() == "/velodyne_points") {
			PointCloudT::ConstPtr cloud = m.instantiate<PointCloudT>();
			assert(cloud);

			cout << cloud->points.size() << endl;

			++frames_read;

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloudColor(cloud, 255, 255, 255);
			viz.addPointCloud(cloud, cloudColor, "cloud");

			viz.spinOnce();
		}
		/*else if (m.getTopic() == "/image_raw") {

		}*/
	}

	cout << "frames read: " << frames_read << endl;

	viz.spin();

	return 0;
}
