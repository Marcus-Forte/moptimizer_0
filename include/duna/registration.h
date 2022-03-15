#pragma once

#include "duna/generic_optimizator.h"
#include "duna/cost/registration_cost.hpp" // TODO can we decouple ?

#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

namespace duna
{
    template <int NPARAM, typename PointSource, typename PointTarget>
    class Registration : public GenericOptimizator<NPARAM>
    {
    public:
        using VectorN = typename GenericOptimizator<NPARAM>::VectorN;
        using DatasetType = typename RegistrationCost<NPARAM, PointSource, PointTarget>::dataset_t;
        using Optimizator<NPARAM>::m_cost;

        Registration(CostFunction<NPARAM> *cost);

        virtual ~Registration();

        inline void setMaxCorrespondenceDistance(float max_dist) { m_max_corr_dist = max_dist; }
        inline void setMaxIcpIterations(unsigned int max_it) { m_icp_iterations = max_it; }
        inline void setCorrespondenceDistanceDamping(float damp) { m_damping_factor = damp;}

        OptimizationStatus minimize(Eigen::Matrix4f &x0_matrix);
        OptimizationStatus minimize(VectorN &x0) override;
        OptimizationStatus minimize();

        Eigen::Matrix4f getFinalTransformation() const;

    protected:
        // Parameters
        unsigned int m_icp_iterations = 20;
        float m_max_corr_dist = 1.0;
        unsigned int m_k_neighboors = 1;

        // Dists damping
        float m_damping_factor = 0.98;
        float m_damping_dists = 1;

        // data
        DatasetType *l_dataset;
        pcl::CorrespondencesPtr m_correspondences;
        typename pcl::PointCloud<PointSource>::Ptr m_source_transformed;
        Eigen::Matrix4f m_final_transform;

        void update_correspondences();
        OptimizationStatus registration_loop();
    };
}