#include "didi_pipeline/utils.h"


namespace didi
{

void calculatePlanePitchAndRoll(ModelCoefficients coefs, float &pitch, float &roll)
{
	assert(coefs.values.size() == 4 && "Plane must have 4 coefs");

	// rotation around Y axis
	pitch = std::atan2(coefs.values[0], coefs.values[2]);

	// rotation around X axis
	roll = std::atan2(coefs.values[1], coefs.values[2]);
}

void colorizeClusters(PointCloudT::Ptr &cloud, const std::vector<pcl::PointIndices> &clusterIndices, PointCloudT::Ptr colorClusters)
{
	for (int ci {}; ci < clusterIndices.size(); ++ci) {
		for (int pi {}; pi < clusterIndices[ci].indices.size(); ++pi) {
			PointT p = cloud->points[clusterIndices[ci].indices[pi]];
			p.intensity = ci;

			colorClusters->points.push_back(p);
		}
	}

	colorClusters->header = cloud->header;
}

};
