/*
 * segmentation.h
 *
 *  Created on: Apr 30, 2017
 *      Author: kaliev
 */

#ifndef DIDI_PIPELINE_SEGMENTATION_H_
#define DIDI_PIPELINE_SEGMENTATION_H_

#include "typedefs.h"
#include "occupancy_grid.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/octree/octree.h>
#include <pcl/search/kdtree.h>

namespace didi
{

using namespace pcl;

void extractEuclideanClusters(
      const PointCloudT::ConstPtr &cloud,
	  octree::OctreePointCloudSearch<PointT>::Ptr tree,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster = 1,
	  unsigned int max_pts_per_cluster = std::numeric_limits<int>::max());

void extractEuclideanClusters(
      const PointCloudT::ConstPtr &cloud,
	  search::KdTree<PointT>::Ptr tree,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster = 1,
	  unsigned int max_pts_per_cluster = std::numeric_limits<int>::max());

void extractEuclideanClusters(
      const PointCloudT::ConstPtr &cloud,
	  OccupancyGrid::Ptr grid,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster = 1,
	  unsigned int max_pts_per_cluster = std::numeric_limits<int>::max());

};

#endif /* DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_SEGMENTATION_H_ */
