#include "didi_pipeline/typedefs.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <assert.h>
#include <string>


using namespace std;
using namespace pcl;
using namespace didi;


static class Node *gInst{};


class Node
{
public:
	ros::Publisher pub;
	Node()
	{
		gInst = this;

		// *** subscribe ***

		velodyne_points_sub_ = nh_.subscribe<PointCloudT>("/velodyne_points", 1, processVelodyne);

		ped_pos_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/obstacle/ped/pose", 1, processPedPose);

		pub = nh_.advertise<visualization_msgs::Marker>("ped", 1);

		// *** advertise ***

		//ped_ref_pos_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/ped_pose_ref", 1);
	}

	int run(int hz = 30)
	{
		ros::Rate loopRate(hz);

		tf2_ros::Buffer tfBuffer;
		tf2_ros::TransformListener tfListener(tfBuffer);

		while (nh_.ok())
		{
			ros::spinOnce();

			try{
				worldVeloTf = tfBuffer.lookupTransform("velodyne", "map", ros::Time(0));
			}
		    catch (tf2::TransformException &ex) {
				ROS_WARN("%s",ex.what());
				ros::Duration(1.0).sleep();
				continue;
		    }

//			visualization_msgs::Marker marker;
//			marker.header.frame_id = "/velodyne";
//			marker.header.stamp = ros::Time::now();
//
//			marker.ns = "bboxes";
//			marker.id = 777;
//
//			marker.type = visualization_msgs::Marker::CUBE;
//			marker.action = visualization_msgs::Marker::ADD;
//
//			marker.lifetime = ros::Duration(0.1);
//
//			marker.pose.position.x = worldVeloTf.transform.translation.x;
//			marker.pose.position.y = worldVeloTf.transform.translation.y;
//			marker.pose.position.z = worldVeloTf.transform.translation.z;
//			marker.pose.orientation.x = 0;//worldVeloTf.transform.rotation.x;
//			marker.pose.orientation.y = 0;//worldVeloTf.transform.rotation.y;
//			marker.pose.orientation.z = 0;//worldVeloTf.transform.rotation.z;
//			marker.pose.orientation.w = 1;//worldVeloTf.transform.rotation.w;
//			marker.scale.x = 1;
//			marker.scale.y = 1;
//			marker.scale.z = 1;
//			marker.color.r = 0.f;
//			marker.color.g = 1.f;
//			marker.color.b = 0.f;
//			marker.color.a = 0.5f;
//
//			pub.publish(marker);

			loopRate.sleep();
		}

		return 0;
	}

	static void processVelodyne(const PointCloudT::ConstPtr &inputCloud)
	{
		//cout << inputCloud->points.size() << endl;
	}

	static void processPedPose(const geometry_msgs::PoseStamped::ConstPtr &worldPed)
	{
		geometry_msgs::TransformStamped worldVeloTf = gInst->worldVeloTf;

		if (worldVeloTf.child_frame_id.empty()) {
			return;
		}

		geometry_msgs::TransformStamped veloPedTf;
		veloPedTf.header.stamp = ros::Time::now();
		veloPedTf.header.frame_id = "velodyne";
		veloPedTf.child_frame_id = "ped_cent2";

		tf2::Vector3 wPedP(worldPed->pose.position.x, worldPed->pose.position.y, worldPed->pose.position.z);

		tf2::Vector3 wVelP(worldVeloTf.transform.translation.x, worldVeloTf.transform.translation.y, worldVeloTf.transform.translation.z);
		tf2::Quaternion wVelQ(worldVeloTf.transform.rotation.x, worldVeloTf.transform.rotation.y, worldVeloTf.transform.rotation.z, worldVeloTf.transform.rotation.w);

		wPedP = tf2::quatRotate(wVelQ, wPedP);
		tf2::Vector3 velPed = wVelP + wPedP;

		tf2::Vector3 pedPed(0., 0., -1.708/2);
		velPed = velPed + pedPed;

		veloPedTf.transform.translation.x = velPed.x();
		veloPedTf.transform.translation.y = velPed.y();
		veloPedTf.transform.translation.z = velPed.z();
		veloPedTf.transform.rotation.x = 0;
		veloPedTf.transform.rotation.y = 0;
		veloPedTf.transform.rotation.z = 0;
		veloPedTf.transform.rotation.w = 1;

		static tf2_ros::TransformBroadcaster tfBroad;
		tfBroad.sendTransform(veloPedTf);
	}

	ros::NodeHandle nh_;

	ros::Subscriber velodyne_points_sub_;

	ros::Subscriber ped_pos_sub_;

	//ros::Publisher ped_ref_pos_pub_;


	geometry_msgs::TransformStamped worldVeloTf;
};


void play_rosbag(string filename)
{
	rosbag::Bag bag;
	bag.open(filename, rosbag::bagmode::Read);

	vector<string> topics;
	topics.push_back("/velodyne_points");

	rosbag::View view(bag, rosbag::TopicQuery(topics));

	int frames_read{};
	bool once{false};

	pcl::visualization::PCLVisualizer viz;

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
}


int main(int argc, char **argv)
{
	string filename("/data/didi/dataset_3/pedestrian/ped_train_8_1495233690_1495233704.bag");

	bool as_node = true;

	if (as_node) {
		const std::string nodeName("ped_refine");

		ros::init(argc, argv, nodeName);

		Node node;

		return node.run();
	}

	return 0;
}
