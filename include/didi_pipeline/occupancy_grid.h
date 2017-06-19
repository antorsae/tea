#ifndef DIDI_PIPELINE_OCCUPANCY_GRID_
#define DIDI_PIPELINE_OCCUPANCY_GRID_

#include "typedefs.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


namespace didi
{


class OccupancyGrid
{
public:
	typedef boost::shared_ptr<OccupancyGrid> Ptr;

	class Cell
	{
	public:
		Cell() {}

		void update(int index, const PointT &p)
		{
			minZ = std::min(minZ, p.z);
			maxZ = std::max(maxZ, p.z);

			indices.push_back(index);
			//cloud_.push_back(p);
		}

		float getOccupValue() const
		{
			return std::max(0.f, maxZ - minZ);
			//return cloud_.size();
		}

		//const PointCloudT& getCloud() const { return cloud_; }

		//PointCloudT cloud_;
		std::vector<int> indices;

		float minZ {1e12};
		float maxZ {-1e12};
	};

	OccupancyGrid() {};

	~OccupancyGrid() {};

	void initGrid(float gridSize = 60, float cellSize = 0.15f);

	void setInputCloud(const PointCloudT::ConstPtr &inputCloud) { cloud_ = inputCloud; }

	void build();

	int radiusSearch(int index, const double radius, std::vector<int> &k_indices,
				                      std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;

	const std::vector<Cell>& getGridCells() { return cells_; }

	void getGridCellsSize(unsigned int &rows, unsigned int &cols) const { rows = gridCols__; cols = gridCols__; }

	PointCloudT::ConstPtr getInputCloud() const { return cloud_; }

	void rasterize(std::vector<uint8_t> &data) const;

private:
	inline bool isIn(const PointT &p) const;

	// convert Cartesian ego coordinates to grid cell's coordinates (row and column)
	inline void cvtEgo2Grid(const PointT &p, int &row, int &col) const;

	// convert Cartesian ego coordinates to grid cell's linear index (row * width + columns)
	inline int cvtEgo2Grid(const PointT &p) const;

private:
	// size of the grid, m
	float gridSize_ {};

	// size of a grid cell, m
	float cellSize_ {};

	std::vector<Cell> cells_;

	// number of columns (and rows) in the grid
	int gridCols__ {};

	float maxOccupValue__ {};

	PointCloudT::ConstPtr cloud_;
};


};


#endif
