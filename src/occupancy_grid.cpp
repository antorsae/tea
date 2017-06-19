#include "didi_pipeline/occupancy_grid.h"

#include <stdio.h>


namespace didi
{


void OccupancyGrid::initGrid(float gridSize, float cellSize)
{
	assert(gridSize > 0 &&
			cellSize > 0 &&
			gridSize > cellSize);

	gridSize_ = gridSize;
	cellSize_ = cellSize;

	const int N = int(gridSize_ / cellSize_);

	printf("Grid size: %ix%i\n", N, N);

	gridCols__ = N;
}

void OccupancyGrid::build()
{
	assert(cloud_ && "Point cloud is not set");
	assert(gridCols__ && "Grid is not initialized");

	cells_ = std::vector<Cell>(gridCols__ * gridCols__);

	maxOccupValue__ = 0.f;

	for (int pi{}; pi < (int)cloud_->points.size(); ++pi) {
		PointT pnt = cloud_->points[pi];

		// skip points that are beyond the grid's boundaries
		if (!isIn(pnt)) {
			continue;
		}
		const int cellIdx = cvtEgo2Grid(pnt);

		cells_[cellIdx].update(pi, pnt);

		maxOccupValue__ = std::max(maxOccupValue__, cells_[cellIdx].getOccupValue());
	}

	int max_pnt_cell {};
	for (const Cell &c : cells_)
		max_pnt_cell = std::max(max_pnt_cell, (int)c.indices.size());
	printf("max cell pnt %i\n", max_pnt_cell);
}

void OccupancyGrid::rasterize(std::vector<uint8_t>& data) const
{
	const int nCells = gridCols__ * gridCols__;

	assert(nCells && (int)cells_.size() == nCells && "Grid is not initialized");
	assert(maxOccupValue__ > 0.f && "Max occupancy value is not valid");

	data.resize(nCells);

	const bool logVal = true; // if true, ignore binarization
	const bool binarize = true;
	const float ocupMin = 0.15f;
	const float ocupMax = 3.f;

	for (int ic {}; ic < nCells; ++ic) {
		float ocupVal = cells_[ic].getOccupValue();
		assert(ocupVal >= 0.f);

		if (binarize) {
			data[ic] = ocupVal >= 0.15f && ocupVal < ocupMax ? 255 : 0;
		}
		else {
			float normVal =	logVal ? std::log(ocupVal+1) / std::log(maxOccupValue__+1) :
							cells_[ic].getOccupValue() / maxOccupValue__;
			assert(normVal >= 0.f);
			data[ic] = uint8_t(255 * normVal);
		}
	}
}

inline bool OccupancyGrid::isIn(const PointT &p) const
{
	const float hs = gridSize_/2;
	return p.x >= -hs && p.x < hs &&
			p.y >= -hs && p.y < hs;
}

int OccupancyGrid::radiusSearch(int index, const double radius,
		std::vector<int>& k_indices, std::vector<float>& k_sqr_distances, unsigned int max_nn) const
{
	assert(cloud_ && "Input point cloud is not set");
	assert(index > -1 && index < (int)cloud_->points.size() && "Invalid point index");

	PointT qPoint = cloud_->points[index];

	if (!isIn(qPoint)) {
		return 0;
	}

	int qRow{}, qCol{};
	cvtEgo2Grid(qPoint, qRow, qCol);

	const int nCells = (radius / cellSize_) + int(int(radius / cellSize_) * cellSize_ < radius);

	for (int row = std::max(0, qRow - nCells); row < std::min(gridCols__ - nCells, qRow + nCells + 1); ++row) {
		for (int col = std::max(0, qCol - nCells); col < std::min(gridCols__ - 1, qCol + nCells + 1); ++col) {
			int cellIdx = col + gridCols__ * row;

			for (int idx : cells_[cellIdx].indices) {
				PointT p = cloud_->points[idx];

				float dx = p.x - qPoint.x, dy = p.y - qPoint.y, dz = p.z - qPoint.z;
				float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

				if (dist <= radius) {
					k_indices.push_back(idx);
					k_sqr_distances.push_back(dist);
				}
			}
		}
	}

	return (int)k_indices.size();
}

inline void OccupancyGrid::cvtEgo2Grid(const PointT& pnt, int& row, int& col) const
{
	col = int(pnt.x / cellSize_) + gridCols__/2;
	row = gridCols__/2 - int(pnt.y / cellSize_);
	assert(row < gridCols__ && col < gridCols__);
}

inline int OccupancyGrid::cvtEgo2Grid(const PointT& pnt) const
{
	int row{}, col{};
	cvtEgo2Grid(pnt, row, col);

	return col + gridCols__ * row;
}


};
