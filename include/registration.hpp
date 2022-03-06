#pragma once

#include "generic_optimizator.h"
#include "cost/registration_cost.hpp" // for dataset type

#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

template <int NPARAM,typename PointSource, typename PointTarget>
class Registration : public GenericOptimizator<NPARAM>
{
public:
    using VectorN = typename GenericOptimizator<NPARAM>::VectorN;
    using Optimizator<NPARAM>::m_cost;

    Registration(CostFunction<NPARAM> *cost) : GenericOptimizator<NPARAM>(cost)
    {
        m_source_transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        m_correspondences = pcl::make_shared<pcl::Correspondences>();

        RegistrationCost<NPARAM,PointSource,PointTarget> *l_cost = reinterpret_cast<RegistrationCost<NPARAM,PointSource,PointTarget> *>(cost);
        l_cost->setCorrespondencesPtr(m_correspondences);
        l_cost->setTransformedSourcePtr(m_source_transformed);
        l_dataset = reinterpret_cast<typename RegistrationCost<NPARAM,PointSource,PointTarget>::datatype_t *>(cost->getDataset());
    }

    virtual ~Registration()
    {
    }

    inline void setMaxCorrespondenceDistance(float max_dist) { m_max_corr_dist = max_dist; }
    inline void setMaxIcpIterations(unsigned int max_it) { m_icp_iterations = max_it; }

    opt_status minimize(VectorN &x0) override
    {

        m_final_transform = Eigen::Matrix4f::Identity();

        Eigen::Matrix4f init_transform;
        so3::param2Matrix6DOF(x0, init_transform);

        pcl::transformPointCloud(*l_dataset->source, *m_source_transformed, init_transform);

        // Perform optimization @ every ICP iteration
        for (int i = 0; i < m_icp_iterations; ++i)
        {
            DUNA_DEBUG_STREAM("## ICP ITERATION: " << i + 1 << "/" << m_icp_iterations << " ##\n");
            update_correspondences();

            opt_status status = GenericOptimizator<NPARAM>::minimize(x0);

            Eigen::Matrix4f delta_transform;
            so3::param2Matrix6DOF(x0, delta_transform);

            pcl::transformPointCloud(*m_source_transformed, *m_source_transformed, delta_transform);
            x0.setZero();

            m_final_transform = delta_transform * m_final_transform;

            if (status == opt_status::SMALL_DELTA)
                return status;
        }

        return opt_status::MAX_IT_REACHED;
    }

    Eigen::Matrix4f getFinalTransformation() const
    {
        return m_final_transform;
    }

protected:
    // Parameters
    unsigned int m_icp_iterations = 20;
    float m_max_corr_dist = 1.0;
    unsigned int m_k_neighboors = 5;

    // data
    typename RegistrationCost<NPARAM,PointSource,PointTarget>::datatype_t *l_dataset;
    pcl::CorrespondencesPtr m_correspondences;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_source_transformed;
    Eigen::Matrix4f m_final_transform;

    void update_correspondences()
    {
        m_correspondences->clear();
        m_correspondences->reserve(m_source_transformed->size());

        pcl::Indices indices(m_k_neighboors);
        std::vector<float> k_distances(m_k_neighboors);

        // compute correspondences
        for (int i = 0; i < m_source_transformed->size(); ++i)
        {

            const pcl::PointXYZ &pt_warped = m_source_transformed->points[i];

            l_dataset->tgt_kdtree->nearestKSearchT(pt_warped, m_k_neighboors, indices, k_distances);

            if (k_distances[0] > m_max_corr_dist * m_max_corr_dist)
                continue;

            // Compute normal

            pcl::Correspondence correspondence;
            correspondence.index_match = indices[0];
            correspondence.index_query = i;
            m_correspondences->push_back(correspondence);
        }

        DUNA_DEBUG("source pts : %ld, corr pts: %ld\n", l_dataset->source->size(), m_correspondences->size());

        if (m_correspondences->size() == 0)
        {
            throw std::runtime_error("no more correspondences.");
        }
    }
};