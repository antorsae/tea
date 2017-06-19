/*
 * typedefs.h
 *
 *  Created on: May 9, 2017
 *      Author: kaliev
 */

#ifndef DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_TYPEDEFS_H_
#define DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_TYPEDEFS_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


namespace didi
{

using namespace pcl;


typedef PointXYZI PointT;
typedef PointCloud<PointT> PointCloudT;


// all quantities in meters
#define CAR_MAX_WIDTH 2.5f
#define CAR_MAX_HEIGHT 2.5f
#define CAR_MAX_LENGTH 5.5f

};

#endif /* DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_TYPEDEFS_H_ */
