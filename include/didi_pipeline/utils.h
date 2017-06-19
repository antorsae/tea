/*
 * utils.h
 *
 *  Created on: May 4, 2017
 *      Author: kaliev
 */

#ifndef DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_UTILS_H_
#define DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_UTILS_H_

#include "typedefs.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>

#include <math.h>


#define CLOCK(f) \
	do { \
	clock_t s = clock(); \
	f; \
	std::cerr << #f": " << 1000.f*(clock() - s)/CLOCKS_PER_SEC << " ms" << std::endl; \
	} while (0)


#define GEN_COLOR(i) ((i + 127) * 2) % 255, ((i + 127) * 3) % 255, ((i + 127) * 5) % 255
#define GEN_COLOR1(i) (((i + 127) * 2) % 255)/255., (((i + 127) * 3) % 255)/255., (((i + 127) * 5) % 255)/255.


namespace didi
{

using namespace pcl;


void calculatePlanePitchAndRoll(ModelCoefficients coefs, float &pitch, float &roll);

void colorizeClusters(PointCloudT::Ptr &cloud, const std::vector<PointIndices> &clusterIndices, PointCloudT::Ptr colorClusters);
//void colorizeClusters(const std::vector<PointCloudT::Ptr> &clusters, PointCloudT::Ptr colorClusters);

};


#endif /* DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_UTILS_H_ */
