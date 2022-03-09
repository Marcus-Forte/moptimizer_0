#pragma once

#include "generic_optimizator.h"
#include "duna/cost/registration_cost.hpp" // for dataset type

#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

template <int NPARAM,typename PointSource, typename PointTarget>
class Registration : public GenericOptimizator<NPARAM>
{
public:
    using Status = typename Optimizator<NPARAM>::Status;
    using VectorN = typename GenericOptimizator<NPARAM>::VectorN;
    using DatasetType = typename RegistrationCost<NPARAM,PointSource,PointTarget>::dataset_t;
    using Optimizator<NPARAM>::m_cost;

    Registration(CostFunction<NPARAM> *cost);

    virtual ~Registration();

    inline void setMaxCorrespondenceDistance(float max_dist) { m_max_corr_dist = max_dist; }
    inline void setMaxIcpIterations(unsigned int max_it) { m_icp_iterations = max_it; }


    Status minimize(Eigen::Matrix4f &x0_matrix);
    Status minimize(VectorN &x0) override;
    Status minimize();
    

    Eigen::Matrix4f getFinalTransformation() const;

protected:
    // Parameters
    unsigned int m_icp_iterations = 20;
    float m_max_corr_dist = 1.0;
    unsigned int m_k_neighboors = 5;

    // data
    DatasetType *l_dataset;
    pcl::CorrespondencesPtr m_correspondences;
    typename pcl::PointCloud<PointSource>::Ptr m_source_transformed;
    Eigen::Matrix4f m_final_transform;

    void update_correspondences();
    Status registration_loop();
};