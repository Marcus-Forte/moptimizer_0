#pragma once

#include "generic_optimizator.h"
#include "duna/cost/registration_cost.hpp" // for dataset type

#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

template <int NPARAM>
class Registration : public GenericOptimizator<NPARAM>
{
public:
    using VectorN = typename GenericOptimizator<NPARAM>::VectorN;
    using Optimizator<NPARAM>::m_cost;

    Registration(CostFunction<NPARAM> *cost);

    virtual ~Registration();

    inline void setMaxCorrespondenceDistance(float max_dist) { m_max_corr_dist = max_dist; }
    inline void setMaxIcpIterations(unsigned int max_it) { m_icp_iterations = max_it; }

    opt_status minimize(VectorN &x0) override;

    Eigen::Matrix4f getFinalTransformation() const;

protected:
    // Parameters
    unsigned int m_icp_iterations = 20;
    float m_max_corr_dist = 1.0;
    unsigned int m_k_neighboors = 5;

    // data
    reg_cost_data_t *l_dataset;
    pcl::CorrespondencesPtr m_correspondences;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_source_transformed;
    Eigen::Matrix4f m_final_transform;

    void update_correspondences();
};