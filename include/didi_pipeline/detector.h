/*
 * detector.h
 *
 *  Created on: May 11, 2017
 *      Author: kaliev
 */

#ifndef DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_DETECTOR_H_
#define DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_DETECTOR_H_


#include "typedefs.h"

#include <pcl/PointIndices.h>
#include <pcl/common/centroid.h>

#include <string>


namespace didi
{


/* Oriented bounding box class */
class OrientedBB
{
public:
	OrientedBB() {}

	Eigen::Quaternionf rotation; // rotation relative to the point cloud frame
	Eigen::Vector3f centroid; // geometrical center of the bb
	float length{}; // x-extent
	float width{}; // y-extent
	float height{}; // z-extent
};


/* Calculate oriented bounding box enclosing the point cloud */
void calculateCarOrientedBB(PointCloudT::Ptr cloud,
		PointCloudT::Ptr projCloud,
		const pcl::PointIndices &indices,
		Eigen::Vector3f upVector,
		OrientedBB &bb);

/* Filter bounding boxes that are likely cars */
void coarseBBFilterCars(const std::vector<OrientedBB> &bbs, std::vector<int> &indices);


/* Track of an object */
class Track
{
public:
	class Node
	{
	public:
		Node(PointCloudT::Ptr cloud_, OrientedBB bb_)
			: cloud(cloud_), bb(bb_)
		{
			precompute();
		}

		Eigen::Vector3f centroid() const
		{
			return bb.centroid;
			//return cloudCentroid;
		}

		PointCloudT::Ptr cloud;

		OrientedBB bb;

		Eigen::Vector3f cloudCentroid;

		double stamp{}; // seconds

	private:
		void precompute()
		{
			// CENTROID
			Eigen::Vector4f v4;
			compute3DCentroid(*cloud, v4);
			cloudCentroid = v4.head<3>();

			// TIMESTAMP
			stamp = cloud->header.stamp * 1e-6;
		}
	};

	Track() {}

	void addNode(const Node &node)
	{
		nodes.push_back(node);
	}

	float distance(const Node &node)
	{
		Eigen::Vector3f d = nodes.back().centroid() - node.centroid();

		return std::sqrt(d.dot(d));
	}

	float distanceExpected(const Node &node)
	{
		if (nodes.size() < 2) {
			return distance(node);
		}

		Eigen::Vector3f ep = 2 * nodes[nodes.size() - 1].centroid() - nodes[nodes.size() - 2].centroid();
		Eigen::Vector3f d = ep - node.centroid();

		return std::sqrt(d.dot(d));
	}

	std::vector<Node> nodes;
};


};


#endif /* DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_DETECTOR_H_ */
