/*
 * tracklet.h
 *
 *  Created on: May 19, 2017
 *      Author: kaliev
 */

#ifndef DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_TRACKLET_H_
#define DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_TRACKLET_H_

#include <Eigen/Dense>

#include <string>
#include <vector>
#include <fstream>


namespace didi
{


class Tracklet
{
public:
	struct Pose
	{
		Eigen::Vector3f t; // translation
		Eigen::Vector3f r; // rotation

		Pose() {}
		Pose(Eigen::Vector3f t_, Eigen::Vector3f r_) : t(t_), r(r_) {}
	};

	Tracklet(std::string objectType_, float width_, float height_, float length_, int firstFrame_ = 0)
		: objectType(objectType_), width(width_), height(height_), length(length_), firstFrame(firstFrame_) {}

	std::string objectType;

	float width;
	float height;
	float length;

	int firstFrame;

	std::vector<Pose> poses;

	int writeXml(std::ofstream &f, int classId, int tabLevel = 0, int firstFrameShift = 0) const;
};

void writeTrackletsXml(std::string filename, const std::vector<Tracklet> &tlets, int firstFrameShift = 0);


};

#endif /* DIDI_PIPELINE_INCLUDE_DIDI_PIPELINE_TRACKLET_H_ */
