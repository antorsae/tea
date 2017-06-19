/*
 * filters.h
 *
 *  Created on: May 1, 2017
 *      Author: kaliev
 */

#ifndef DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_FILTERS_H_
#define DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_FILTERS_H_

#include "typedefs.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>


namespace didi
{

using namespace pcl;

// cut points inside volume defined by internal and external bounding boxes
void cutVolume(const PointCloudT::ConstPtr &inputCloud, PointCloudT::Ptr outputCloud,
		PointXYZ inMin, PointXYZ inMax, PointXYZ exMin, PointXYZ exMax);


void removeGroundPoints(const PointCloudT::ConstPtr &inputCloud,
		PointCloudT::Ptr outputCloud, ModelCoefficients *outPlaneCoefs = nullptr);


void projectPointsOnPlane(const PointCloudT::ConstPtr &inputCloud, PointCloudT::Ptr outputCloud,
		ModelCoefficients planeCoefs);

}


#endif /* DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_FILTERS_H_ */
