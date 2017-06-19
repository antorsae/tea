#include "didi_pipeline/filters.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>


namespace didi
{

void cutVolume(const PointCloudT::ConstPtr &inputCloud, PointCloudT::Ptr outputCloud,
		PointXYZ inMin, PointXYZ inMax, PointXYZ exMin, PointXYZ exMax)
{
	outputCloud->header = inputCloud->header;
	outputCloud->is_dense = inputCloud->is_dense;
	for (const PointXYZI &p : inputCloud->points) {
		// must be outside internal bbox and inside external bbox
		bool outIn = p.x <= inMin.x || p.x >= inMax.x || p.y <= inMin.y || p.y >= inMax.y || p.z <= inMin.z || p.z >= inMax.z;
		bool inEx = p.x > exMin.x && p.x < exMax.x && p.y > exMin.y && p.y < exMax.y && p.z > exMin.z && p.z < exMax.z;
		if (outIn && inEx) {
			outputCloud->points.push_back(p);
		}
	}
}

// TODO move to filters.h
void removeGroundPoints(const PointCloudT::ConstPtr &inputCloud,
		PointCloudT::Ptr outputCloud, pcl::ModelCoefficients *outPlaneCoefs)
{
	// using RANSAC algorithm fit a plane to the point cloud and extract the outliers

	// tune these params
	const double distThresh {0.2}; // meters
	const int maxIters {100};

	pcl::ModelCoefficients coefs;
	pcl::PointIndicesPtr inliers(new pcl::PointIndices);

	// TODO compare methods (RANSAC, MLESAC, LMEDS, ...)

	// setup RANSAC
	pcl::SACSegmentation<PointXYZI> seg;
	//seg.setOptimizeCoefficients(true); // optional
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(maxIters);
	seg.setDistanceThreshold(distThresh);

	// fit a plane
	seg.setInputCloud(inputCloud);
	seg.segment(*inliers, coefs);

	if (outPlaneCoefs) {
		*outPlaneCoefs = coefs;
	}

	// extract the outliers
	pcl::ExtractIndices<PointXYZI> extract;
	extract.setInputCloud(inputCloud);
	extract.setIndices(inliers);
	extract.setNegative(true);
	extract.filter(*outputCloud);
}

void projectPointsOnPlane(const PointCloudT::ConstPtr &inputCloud, PointCloudT::Ptr outputCloud,
		ModelCoefficients planeCoefs)
{
	assert(planeCoefs.values.size() == 4);
	Eigen::Vector3f n3(planeCoefs.values[0], planeCoefs.values[1], planeCoefs.values[2]);
	Eigen::Vector4f n4(planeCoefs.values[0], planeCoefs.values[1], planeCoefs.values[2], planeCoefs.values[3]);

	for (PointXYZI p_ : inputCloud->points) {
		Eigen::Vector3f p3(p_.x, p_.y, p_.z);
		Eigen::Vector4f p4(p_.x, p_.y, p_.z, 1.f);

		Eigen::Vector3f proj = p3 - n4.dot(p4) * n3;
		//std::cerr << d << std::endl;

		PointXYZI proj_;
		proj_.x = proj.x();
		proj_.y = proj.y();
		proj_.z = proj.z();
		proj_.intensity = p_.intensity;
		outputCloud->points.push_back(proj_);
	}

	outputCloud->header = inputCloud->header;
	outputCloud->is_dense = inputCloud->is_dense;
}

}
