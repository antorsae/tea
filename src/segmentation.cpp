#include "didi_pipeline/segmentation.h"


namespace didi
{

class PointSearchBase
{
public:
	virtual ~PointSearchBase() {}

	virtual int radiusSearch(int index, const double radius, std::vector<int> &k_indices,
	                      std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const = 0;

	virtual PointCloudT::ConstPtr getInputCloud() const = 0;
};

class PointSearchOctree : public PointSearchBase
{
public:
	PointSearchOctree(octree::OctreePointCloudSearch<PointT>::Ptr tree) : tree_(tree) {}

	int radiusSearch(int index, const double radius, std::vector<int> &k_indices,
		                      std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
	{
		return tree_->radiusSearch(index, radius, k_indices, k_sqr_distances, max_nn);
	}

	PointCloudT::ConstPtr getInputCloud() const
	{
		return tree_->getInputCloud();
	}

private:
	octree::OctreePointCloudSearch<PointT>::Ptr tree_;
};

class PointSearchKdTree : public PointSearchBase
{
public:
	PointSearchKdTree(search::KdTree<PointT>::Ptr tree) : tree_(tree) {}

	int radiusSearch(int index, const double radius, std::vector<int> &k_indices,
		                      std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
	{
		return tree_->radiusSearch(index, radius, k_indices, k_sqr_distances, max_nn);
	}

	PointCloudT::ConstPtr getInputCloud() const
	{
		return tree_->getInputCloud();
	}

private:
	search::KdTree<PointT>::Ptr tree_;
};

class PointSearchOccupancyGrid : public PointSearchBase
{
public:
	PointSearchOccupancyGrid(OccupancyGrid::Ptr grid) : grid_(grid) {}

	int radiusSearch(int index, const double radius, std::vector<int> &k_indices,
			                      std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
	{
		return grid_->radiusSearch(index, radius, k_indices, k_sqr_distances, max_nn);
	}

	PointCloudT::ConstPtr getInputCloud() const
	{
		return grid_->getInputCloud();
	}

private:
	OccupancyGrid::Ptr grid_;
};

void extractEuclideanClusters(
      const PointCloudT &cloud,
	  PointSearchBase *ps,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster,
	  unsigned int max_pts_per_cluster)
{
	assert(cloud.points.size() == ps->getInputCloud()->points.size() && "Input cloud and search cloud sizes must match");

	std::vector<bool> processed(cloud.points.size(), false);

	for (int i = 0; i < (int)cloud.points.size(); ++i) {
		if (processed[i])
			continue;

		std::vector<int> seed_queue;
		int sq_idx = 0;
		seed_queue.push_back(i);

		processed[i] = true;

		while (sq_idx < (int)seed_queue.size()) {
			std::vector<int> nn_indices;
			std::vector<float> nn_distances;

			if (tolerance < 0) {
				const float alpha = DEG2RAD(0.4f); // lidar azimuth resolution, deg
				const float min_lateral_ofst = 2.f; // minimal lateral offset btw the host and obstacle car, m
				const float max_car_length = 4.f; // maximum car length (maximum distance between lidar points reflected from a car)

				// approx distance to an obstacle
				PointT pnt = cloud.points[seed_queue[sq_idx]];
				const float r2 = pnt.x * pnt.x + pnt.y * pnt.y + pnt.z * pnt.z;

				tolerance = std::sqrt(r2) < 12.f ? 0.5f
						: alpha * r2 / min_lateral_ofst;

				// tol cannot be larger than the distance btw the two most distant points
				tolerance = std::min(max_car_length, tolerance);
			}

			if (!ps->radiusSearch(seed_queue[sq_idx], tolerance, nn_indices, nn_distances)) {
				sq_idx++;
				continue;
			}

			for (size_t j = 0; j < nn_indices.size(); ++j) {
				if (nn_indices[j] == -1 || processed[nn_indices[j]])
					continue;

				seed_queue.push_back(nn_indices[j]);
				processed[nn_indices[j]] = true;
			}

			sq_idx++;
		}

		if (seed_queue.size() >= min_pts_per_cluster && seed_queue.size() <= max_pts_per_cluster) {
			clusters.push_back(PointIndices());
			PointIndices &pi = clusters.back();
			pi.header = cloud.header;
			pi.indices = seed_queue;

//			pcl::PointIndices r;
//			r.indices.resize(seed_queue.size ());
//			for (size_t j = 0; j < seed_queue.size (); ++j)
//				r.indices[j] = seed_queue[j];
//
//			size_t oldS = r.indices.size();
//			std::sort(r.indices.begin (), r.indices.end ());
//			r.indices.erase(std::unique (r.indices.begin (), r.indices.end ()), r.indices.end());
//			assert(r.indices.size() == oldS);
//
//			r.header = cloud.header;
//			clusters.push_back(r);
		}
	}

//	{
//		int m {};
//		for (auto c : clusters) m = std::max(m, (int)c.indices.size());
//		std::cout << "max cluster sz " << m << std::endl;
//	}
}

void extractEuclideanClusters(
      const PointCloudT::ConstPtr &cloud,
	  octree::OctreePointCloudSearch<PointT>::Ptr tree,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster,
	  unsigned int max_pts_per_cluster)
{
	PointSearchOctree ps(tree);

	extractEuclideanClusters(*cloud, &ps, tolerance, clusters, min_pts_per_cluster, max_pts_per_cluster);
}

void extractEuclideanClusters(
      const PointCloudT::ConstPtr &cloud,
	  search::KdTree<PointT>::Ptr tree,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster,
	  unsigned int max_pts_per_cluster)
{
	PointSearchKdTree ps(tree);

	extractEuclideanClusters(*cloud, &ps, tolerance, clusters, min_pts_per_cluster, max_pts_per_cluster);
}

void extractEuclideanClusters(
	  const PointCloudT::ConstPtr &cloud,
	  OccupancyGrid::Ptr grid,
      float tolerance,
	  std::vector<PointIndices> &clusters,
      unsigned int min_pts_per_cluster,
	  unsigned int max_pts_per_cluster)
{
	PointSearchOccupancyGrid ps(grid);

	extractEuclideanClusters(*cloud, &ps, tolerance, clusters, min_pts_per_cluster, max_pts_per_cluster);
}

};
