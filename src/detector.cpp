#include "didi_pipeline/detector.h"

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>


namespace didi
{


void calculateCarOrientedBB(PointCloudT::Ptr cloud,
		PointCloudT::Ptr projCloud,
		const pcl::PointIndices &indices,
		Eigen::Vector3f upVector,
		OrientedBB &bb)
{
    Eigen::Vector4f centroid;
    compute3DCentroid(*projCloud, indices, centroid);

    // car's local coordinate system:
    // (0)X+ - from back to front / length
    // (1)Y+ - from right to left (if looking along X+) / width
    // (2)Z+ - from ground to sky / height
    Eigen::Matrix3f axes;

    // do eigendecomposition
    Eigen::Matrix3f covariance_matrix;
    computeCovarianceMatrixNormalized(*projCloud, indices, centroid, covariance_matrix);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);

    // X axis would be the eigenvector with the largest eigenvalue, since car's length is the largest dimension.
    axes.col(0) = eigen_solver.eigenvectors().col(2); // eigenvalues are sorted in increasing order

    // Z axis would be the up vector
    axes.col(2) = upVector;

    // Y axis would be the cross product of Z and X (right-handed frame)
    axes.col(1) = axes.col(2).cross(axes.col(0));

    // transform point cloud to car's local frame
    Eigen::Matrix4f localTf(Eigen::Matrix4f::Identity());
    localTf.block<3, 3>(0, 0) = axes.transpose();
    localTf.block<3, 1>(0, 3) = -1.f * (axes.transpose() * centroid.head<3>()); // need to rotate frame offset too
    PointCloudT::Ptr localCloud(new PointCloudT);
    transformPointCloud(*cloud, indices, *localCloud, localTf);

    // get point cloud dimensions
    PointT minP, maxP;
    getMinMax3D(*localCloud, minP, maxP);

    bb.rotation = Eigen::Quaternionf(axes);

    Eigen::Vector3f bbCenter = 0.5f * (minP.getArray3fMap() + maxP.getArray3fMap());
    bb.centroid = axes * bbCenter + centroid.head<3>();

    bb.length = maxP.x - minP.x;
    bb.width = maxP.y - minP.y;
    bb.height = maxP.z - minP.z;
}

void coarseBBFilterCars(const std::vector<OrientedBB> &bbs, std::vector<int> &indices)
{
	assert(indices.empty() && "Indices array must be clear");

	//const float carMaxExtent = std::max(CAR_MAX_WIDTH, std::max(CAR_MAX_HEIGHT, CAR_MAX_LENGTH));

	for (int i{}; i < (int)bbs.size(); ++i) {
		const OrientedBB &bb = bbs[i];

		if (bb.length < CAR_MAX_LENGTH &&
				bb.width < CAR_MAX_WIDTH &&
				bb.height < CAR_MAX_HEIGHT) {
			indices.push_back(i);
		}

		//indices.push_back(i);

//		if (std::max(bb.length, std::max(bb.width, bb.height)) < carMaxExtent) {
//			indices.push_back(i);
//		}
	}
}


};
