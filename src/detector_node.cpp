#include "didi_pipeline/typedefs.h"
#include "didi_pipeline/occupancy_grid.h"
#include "didi_pipeline/segmentation.h"
#include "didi_pipeline/filters.h"
#include "didi_pipeline/utils.h"
#include "didi_pipeline/detector.h"
#include "didi_pipeline/tracklet.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/MarkerArray.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// to cloud filter
#include <pcl/filters/voxel_grid.h>

// to track manager
#include <pcl/common/centroid.h>
#include <list>

#include <string>
#include <numeric>


namespace didi
{

using namespace std;


static class DetectorNode *gInst {nullptr};

class DetectorNode
{
public:
	DetectorNode(int argc, char **argv, std::string nodeName)
	{
		assert(gInst == nullptr && "Only one instance is allowed");
		gInst = this;

		if (argc > 1) {
			bagfilename = string(argv[1]);

			bTest_ = true;

			viewer_ = boost::unique_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer);
			viewer_->addCoordinateSystem(2.f);
		}
		else {
			sub_ = nh_.subscribe<PointCloudT>("velodyne_points", 1, processVelodyne);
		}

		pub_ = nh_.advertise<PointCloudT>("velodyne_points_noground", 1);

		pubClusters_ = nh_.advertise<PointCloudT>("velodyne_points_clusters", 1);

		pubBboxes_ = nh_.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 1);

		pubHeightmap_ = nh_.advertise<sensor_msgs::Image>("heightmap", 1);

		grid_ = boost::shared_ptr<OccupancyGrid>(new OccupancyGrid());
		grid_->initGrid(60, 0.15f);
	}

	int run(int hz = 10)
	{
		if (bTest_) {
			play(bagfilename);
		}
		else {
			ros::Rate loopRate(hz);

			while (nh_.ok())
			{
				ros::spinOnce();

				loopRate.sleep();
			}
		}

		return 0;
	}

	int play(const string &filename)
	{
		rosbag::Bag bag;
		bag.open(filename, rosbag::bagmode::Read);

		vector<string> topics;
		topics.push_back("/image_raw");
		topics.push_back("/velodyne_points");

		rosbag::View view(bag, rosbag::TopicQuery(topics));

		// TODO synchronize messages

		std::vector<double> cameraStamps;

//		int msgs {};
		for (rosbag::MessageInstance const m : view) {
			if (m.getTopic() == "/velodyne_points") {
				PointCloudT::ConstPtr cloud = m.instantiate<PointCloudT>();
				assert(cloud);
				//cout << "v " << cloud->header.stamp * 1e-6 - cameraStamps.back() << endl;
				processPointCloud(cloud);
			}
			else if (m.getTopic() == "/image_raw") {
				sensor_msgs::ImageConstPtr image = m.instantiate<sensor_msgs::Image>();
				assert(image);

				cameraStamps.push_back(image->header.stamp.sec + image->header.stamp.nsec * 1e-9);
				//cout << "i " << cameraStamps.back() << endl;
			}

//			if (++msgs > 50) {
//				break;
//			}
		}

		int c_max{}, t_max{};
		for (int ti{}; ti<tracks_.size(); ++ti)
			if (tracks_[ti].nodes.size() > c_max) {
				c_max = tracks_[ti].nodes.size();
				t_max = ti;
			}
		cout << "longest track " << t_max << endl;

		std::vector<int> chosenTracks{110, 287};

		generateTracklets(cameraStamps, chosenTracks);

		const double magicTimeShift{-0.5356};
		const int camFps{25};
		writeTrackletsXml("tracklets.xml", tracklets_, int(magicTimeShift * camFps));

		visualizeTracks();//chosenTracks);

		viewer_->spin();

		return 0;
	}

private:
	std::vector<Track> tracks_;
	std::list<int> activeTracks_;

	void processPointCloud(const PointCloudT::ConstPtr &inputCloud)
	{
		PointCloudT::Ptr cutVolCloud(new PointCloudT());
		pcl::PointXYZ inMin(-2, -1, -3), inMax(2, 1, 2);
		pcl::PointXYZ exMin(-30, -30, -3), exMax(30, 30, 2);
		didi::cutVolume(inputCloud, cutVolCloud, inMin, inMax, exMin, exMax);

		PointCloudT::Ptr noGndCloud(new PointCloudT());
		pcl::ModelCoefficients planeCoefs;
		didi::removeGroundPoints(cutVolCloud, noGndCloud, &planeCoefs);

		PointCloudT::Ptr filtCloud = noGndCloud;
//		PointCloudT::Ptr filtCloud(new PointCloudT());
//		filterCloud(noGndCloud, filtCloud);

		PointCloudT::Ptr projCloud(new PointCloudT());
		didi::projectPointsOnPlane(filtCloud, projCloud, planeCoefs);

		std::vector<pcl::PointIndices> clustersPointIndices;
		extractClusters(projCloud, clustersPointIndices);

		// get cluster bounding boxes
		std::vector<OrientedBB> bbArr;
		Eigen::Vector3f upVector(planeCoefs.values[0], planeCoefs.values[1], planeCoefs.values[2]);
		for (int i{}; i < clustersPointIndices.size(); ++i) {
			didi::OrientedBB bb;
			calculateCarOrientedBB(filtCloud, projCloud, clustersPointIndices[i], upVector, bb);
			bbArr.push_back(bb);
		}

		std::vector<int> filtClusterIndices;
		didi::coarseBBFilterCars(bbArr, filtClusterIndices);

		if (bTest_) {

			// ---

			std::list<int> activeTracksUpd;

			//cout << "clusters " << clustersPointIndices.size() << endl;

			for (int ci{}; ci < (int)clustersPointIndices.size(); ++ci) { // clusters
			//for (int ci : filtClusterIndices) {
				PointCloudT::Ptr clusterCloud(new PointCloudT);
				for (int pi : clustersPointIndices[ci].indices) {
					clusterCloud->points.push_back(filtCloud->points[pi]);
				}
				clusterCloud->header = filtCloud->header;

				Track::Node node{clusterCloud, bbArr[ci]};

				float minDist {1e6};
				int minDistTrackInd {-1};
				for (int ti : activeTracks_) { // active tracks
					//float dist = tracks_[ti].distance(node);
					float dist = tracks_[ti].distanceExpected(node);
					if (minDist > dist) {
						minDist = dist;
						minDistTrackInd = ti;
					}
				}

				cout << "min dist " << minDist << endl;

				const float distThreshold = 0.1f * (80. * 1000. / 3600.);
				if (minDist < distThreshold) {
					cout << minDistTrackInd << " <- " << ci << endl;
					tracks_[minDistTrackInd].addNode(node);

					// ensure only one node could be added to a track per frame
					activeTracks_.remove(minDistTrackInd);

					activeTracksUpd.push_back(minDistTrackInd);
				}
				else {
					cout << "new track " << tracks_.size() << endl;
					Track track;
					track.addNode(node);
					tracks_.push_back(track);

					activeTracksUpd.push_back(tracks_.size() - 1);
				}
			}

			activeTracks_ = activeTracksUpd;

			// ---

//			PointCloudT::Ptr colorClusters(new PointCloudT);
//			std::vector<pcl::PointIndices> clusterIndicesFiltered;
//			for (int i : bbArrIndices) clusterIndicesFiltered.push_back(clustersPointIndices[i]);
//			colorizeClusters(filtCloud, clusterIndicesFiltered, colorClusters);
//
//			static int pcIdx{};
//			viewer_->addPointCloud<PointT>(colorClusters, "cloud" + std::to_string(pcIdx++));

			viewer_->spinOnce();
		}
		else {
			// ground extraction
			pub_.publish(noGndCloud);

			// clusterization
			PointCloudT::Ptr colorClusters(new PointCloudT);
			colorizeClusters(filtCloud, clustersPointIndices, colorClusters);
			pubClusters_.publish(colorClusters);

			// all bounding boxes
			visualization_msgs::MarkerArray markerArr;
			createMarkers(bbArr, filtClusterIndices, markerArr);
			pubBboxes_.publish(markerArr);

			//		float pitch {}, roll {};
			//		didi::calculatePlanePitchAndRoll(planeCoefs, pitch, roll);
			//		//cerr << "pitch=" << RAD2DEG(pitch) << endl;
			//		cerr <<
			//		std::accumulate(planeCoefs.values.begin(), planeCoefs.values.end()-1, 0.f, [](float a, float b){return a + b*b;})
			//		<< endl;

			//		sensor_msgs::Image hmImage;
			//		hmImage.encoding = "mono8";
			//		grid_->getGridCellsSize(hmImage.height, hmImage.width);
			//		grid_->rasterize(hmImage.data);
			//
			//		pubHeightmap_.publish(hmImage);
		}
	}

	static void processVelodyne(const PointCloudT::ConstPtr &inputCloud)
	{
		gInst->processPointCloud(inputCloud);
	}

	static void filterCloud(const PointCloudT::ConstPtr &inputCloud, PointCloudT::Ptr filtCloud)
	{
		pcl::VoxelGrid<PointT> vg;

		// grid leaf size from Velodyne HDL-32E accuracy
		float ls = 0.1f;

		vg.setInputCloud(inputCloud);
		vg.setLeafSize(ls, ls, ls);
		vg.filter(*filtCloud);
	}

	static void extractClusters(const PointCloudT::ConstPtr &inputCloud, std::vector<pcl::PointIndices> &clusterIndices)
	{
		// 0 - octree (fastest)
		// 1 - kd-tree
		// 2 - occupancy grid
		const bool useMethod {0};

		const float tolerance {1}; // 1 meter is good for car
		const float epsilon {0.02}; // defined by the lidar
		const int minClusterSize {100};
		const int maxClusterSize {10000};

		clock_t start = clock();

		if (useMethod == 0) {
			double octreeResolution = 0.5;
			pcl::octree::OctreePointCloudSearch<PointT>::Ptr tree(
					new pcl::octree::OctreePointCloudSearch<PointT>(octreeResolution));
			//tree->enableDynamicDepth(100); // !! causes crash
			tree->setEpsilon(epsilon);
			tree->setInputCloud(inputCloud);
			tree->addPointsFromInputCloud();

			didi::extractEuclideanClusters(inputCloud,
					tree,
					tolerance,
					clusterIndices,
					minClusterSize,
					maxClusterSize);
		}
		else if (useMethod == 1) {
			const bool sorted {true};
			pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>(sorted));
			tree->setInputCloud(inputCloud);
			tree->setEpsilon(epsilon);

			didi::extractEuclideanClusters(inputCloud,
				tree,
				tolerance,
				clusterIndices,
				minClusterSize,
				maxClusterSize);
		}
		else if (useMethod == 2) {
			OccupancyGrid::Ptr grid = gInst->grid_;

		    grid->setInputCloud(inputCloud);
			grid->build();

			didi::extractEuclideanClusters(inputCloud,
					gInst->grid_,
					tolerance,
					clusterIndices,
					minClusterSize,
					maxClusterSize);
		}

		printf("clusters: %i, time: %f\n", (int)clusterIndices.size(), 1000.*(clock() - start)/CLOCKS_PER_SEC);
		printf("%i\n", inputCloud->points.size());

//		for (pcl::PointIndices indices : clusterIndices)
//		{
//			PointCloudT::Ptr cluster(new PointCloudT);
//			cluster->header = inputCloud->header;
//
//			for (int i : indices.indices) {
//				cluster->points.push_back(inputCloud->points[i]);
//			}
//
//			clusters.push_back(cluster);
//		}
	}

	static void createMarkers(const std::vector<OrientedBB> &bbArr, const std::vector<int> &indices, visualization_msgs::MarkerArray &markerArr)
	{
		std::vector<OrientedBB> masked;
		for (int i : indices) {
			masked.push_back(bbArr[i]);
		}

		createMarkers(masked, markerArr);
	}

	static void createMarkers(const std::vector<OrientedBB> &bbArr, visualization_msgs::MarkerArray &markerArr)
	{
		for (int i{}; i < bbArr.size(); ++i) {
			visualization_msgs::Marker marker;
			marker.header.frame_id = "/velodyne";
			marker.header.stamp = ros::Time::now();

			marker.ns = "bboxes";
			marker.id = i;

			marker.type = visualization_msgs::Marker::CUBE;
			marker.action = visualization_msgs::Marker::ADD;

			marker.lifetime = ros::Duration(0.1);

			didi::OrientedBB bb = bbArr[i];

			marker.pose.position.x = bb.centroid.x();
			marker.pose.position.y = bb.centroid.y();
			marker.pose.position.z = bb.centroid.z();
			marker.pose.orientation.x = bb.rotation.x();
			marker.pose.orientation.y = bb.rotation.y();
			marker.pose.orientation.z = bb.rotation.z();
			marker.pose.orientation.w = bb.rotation.w();
			marker.scale.x = bb.length;
			marker.scale.y = bb.width;
			marker.scale.z = bb.height;
			marker.color.r = 0.f;
			marker.color.g = 1.f;
			marker.color.b = 0.f;
			marker.color.a = 0.5f;

			markerArr.markers.push_back(marker);
		}
	}

	void visualizeTracks(std::vector<int> chosenTracks = std::vector<int>())
	{
		if (chosenTracks.empty()) {
			chosenTracks.resize(tracks_.size());
			std::iota(chosenTracks.begin(), chosenTracks.end(), 0);
		}

		for (int ti : chosenTracks) {
			const Track &track = tracks_[ti];

			const int trackLengthThreshold {10};
			if (track.nodes.size() < trackLengthThreshold) {
				continue;
			}

			Eigen::Vector3f cnt_l;

			const int drawStep{1};

			//for (int ci : vector<int>{23,22}) {
			for (int ci{0}; ci < track.nodes.size(); ci += drawStep) {
				std::string suffix = std::to_string(ti) + "_" + std::to_string(ci);

				pcl::visualization::PointCloudColorHandlerCustom<PointT> cloudColor(track.nodes[ci].cloud, GEN_COLOR(17*ci+1));
				viewer_->addPointCloud<PointT>(track.nodes[ci].cloud, cloudColor, "cloud" + suffix);

				//Eigen::Vector3f cnt = track.nodes[ci].cloudCentroid;
				Eigen::Vector3f cnt = track.nodes[ci].bb.centroid;

				if (ci == 0) {
					cnt_l = cnt;
					continue;
				}

				PointXYZ p1(cnt_l.x(), cnt_l.y(), cnt_l.z());
				PointXYZ p2(cnt.x(), cnt.y(), cnt.z());
				viewer_->addLine(p2, p1, GEN_COLOR1(ti), "track" + suffix, 0);

				viewer_->addSphere(p1, 0.1, "centroid_" + suffix);

				//OrientedBB bb = track.nodes[ci].bb;
				//viewer_->addCube(bb.centorid, bb.rotation, bb.length, bb.width, bb.height, "bb" + suffix);

				//if (ci == track.nodes.size() - 1) {
					viewer_->addText3D("track" + std::to_string(ti) + "_" + std::to_string(ci-1),
							p1, 0.2f, GEN_COLOR1(ti), "label" + std::to_string(ti) + "_" + std::to_string(ci-1));
				//}

				cnt_l = cnt;
			}
		}

		for (int ti{}; ti < (int)tracklets_.size(); ++ti) {
			const Tracklet &tlet = tracklets_[ti];

			Eigen::Vector3f cnt_l;

			for (int pi{}; pi < tlet.poses.size(); ++pi) {
				Eigen::Vector3f cnt = tlet.poses[pi].t;

				if (pi == 0) {
					cnt_l = cnt;
				}

				PointXYZ p1(cnt_l.x(), cnt_l.y(), cnt_l.z());
				PointXYZ p2(cnt.x(), cnt.y(), cnt.z());
				viewer_->addLine(p2, p1, GEN_COLOR1(ti), "tlet" + std::to_string(ti) + "_" + std::to_string(pi), 0);

				//viewer_->addSphere(p1, 0.1, "centroid_" + std::to_string(ti) + "_" + std::to_string(pi));

				cnt_l = cnt;
			}
		}
	}

	void generateTracklets(const std::vector<double> &stamps, std::vector<int> chosenTracks)
	{
		auto printV = [](Eigen::Vector3f v, bool e = false) {
			cout << v.x() << ", " <<  v.y() << ", " << v.z();
			if (e) cout << endl;
		};

		for (int ti : chosenTracks) {
			const Track &track = tracks_[ti];

			// seek the nearest stamp to the left-most track node
			int si{};
			for (; si < (int)stamps.size(); ++si) {
				if (stamps[si] >= track.nodes.front().stamp) {
					/*cout << stamps[si] - 1.49099e+09 << " " << tracks_[ti].nodes.front().stamp - 1.49099e+09
							<< " " << si << " " << ti << endl;*/
					break;
				}
			}

			Tracklet tlet("Car", 1.725, 2., 4.450, si);

			vector<Tracklet::Pose> &poses = tlet.poses;

			vector<Eigen::Vector3f> velHist;

			// interpolation
			for (int ni{}; ni < (int)track.nodes.size() - 1; ++ni) {
				const Track::Node &node0 = track.nodes[ni];
				const Track::Node &node1 = track.nodes[ni + 1];

				Eigen::Vector3f c0 = node0.centroid();
				Eigen::Vector3f c1 = node1.centroid();
				double t0 = node0.stamp;
				double t1 = node1.stamp;

				//cout << setprecision(15) << node0.stamp << "-" << node1.stamp << endl;
				printV(c0);
				cout <<  " - ";
				printV(c1, true);

				while (stamps[si] < node1.stamp) {
					//cout << setprecision(15) << "\t" << stamps[si] << endl;
					Eigen::Vector3f cInt = c0 + (stamps[si] - t0) * (c1 - c0) / (t1 - t0);

					poses.push_back(Tracklet::Pose{cInt, Eigen::Vector3f::Zero()});

					printV(cInt, true);

					velHist.push_back((c1 - c0) / (t1 - t0));

					++si;
				}
			}

			Eigen::Vector3f meanVelBwd;
			Eigen::Vector3f meanVelFwd;
			const float tau {0.99};

			for (int i{}; i < velHist.size(); ++i) {
				meanVelFwd = (1. - tau) * meanVelFwd + tau * velHist[i];
				meanVelBwd = (1. - tau) * meanVelBwd + tau * velHist[velHist.size() - 1 - i];
			}

			// extrapolation
			float extrFwd{3.0}; // sec
			float extrBwd{3.0};
			float cameraFps{25};
			int extrFwdNum{extrFwd * cameraFps};
			int extrBwdNum{extrBwd * cameraFps};

			vector<Tracklet::Pose> predBwd;
			for (int i{}; i < extrBwdNum - 1; ++i) {
				float dt = (extrBwdNum - 1 - i) * (1. / cameraFps);

				Eigen::Vector3f c0 = poses.front().t;
				Eigen::Vector3f cExtr = c0 - dt * meanVelBwd;

				predBwd.push_back(Tracklet::Pose{cExtr, Eigen::Vector3f::Zero()});
			}

			poses.insert(poses.begin(), predBwd.begin(), predBwd.end());

			tlet.firstFrame -= extrBwdNum - 1;

			vector<Tracklet::Pose> predFwd;
			for (int i{}; i < extrFwdNum; ++i) {
				float dt = i * (1. / cameraFps);

				Eigen::Vector3f cn = poses.back().t;
				Eigen::Vector3f cExtr = cn + dt * meanVelFwd;

				predFwd.push_back(Tracklet::Pose{cExtr, Eigen::Vector3f::Zero()});
			}

			poses.insert(poses.end(), predFwd.begin(), predFwd.end());

			tracklets_.push_back(tlet);
			cout << "tlet has poses: " << tlet.poses.size() << endl;
		}
	}

private:
	string bagfilename;

	ros::NodeHandle nh_;

	ros::Publisher pub_;

	ros::Publisher pubClusters_;

	ros::Publisher pubBboxes_;

	ros::Publisher pubHeightmap_;

	ros::Subscriber sub_;

	OccupancyGrid::Ptr grid_;

	vector<Tracklet> tracklets_;

	boost::unique_ptr<pcl::visualization::PCLVisualizer> viewer_;

	bool bTest_ {}; // test regime
};

}; // namespace didi

int main(int argc, char **argv)
{
	const std::string nodeName("didi_pipeline_detector");

	ros::init(argc, argv, nodeName);

	didi::DetectorNode node(argc, argv, nodeName);

	return node.run();
}
