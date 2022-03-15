#pragma once

#include "duna/registration.h"

namespace duna
{
    template <int NPARAM, typename PointSource, typename PointTarget>
    Registration<NPARAM, PointSource, PointTarget>::Registration(CostFunction<NPARAM> *cost) : GenericOptimizator<NPARAM>(cost)
    {
        m_source_transformed = pcl::make_shared<pcl::PointCloud<PointSource>>();
        m_correspondences = pcl::make_shared<pcl::Correspondences>();

        RegistrationCost<NPARAM, PointSource, PointTarget> *l_cost = reinterpret_cast<RegistrationCost<NPARAM, PointSource, PointTarget> *>(cost);
        l_cost->setCorrespondencesPtr(m_correspondences);
        l_cost->setTransformedSourcePtr(m_source_transformed);
        l_dataset = reinterpret_cast<DatasetType *>(cost->getDataset());
        m_final_transform = Eigen::Matrix4f::Identity(); // TODO move away ?
        // By Default, we want to set the internal optimization loop to a single iteration to allow ICP to transform source more often.
        Optimizator<NPARAM>::setMaxOptimizationIterations(1);
    }

    template <int NPARAM, typename PointSource, typename PointTarget>
    Registration<NPARAM, PointSource, PointTarget>::~Registration()
    {
    }

    template <int NPARAM, class PointSource, typename PointTarget>
    OptimizationStatus Registration<NPARAM, PointSource, PointTarget>::minimize()
    {
        m_final_transform = Eigen::Matrix4f::Identity();
        OptimizationStatus status = registration_loop();

        return status;
    }

    template <int NPARAM, class PointSource, typename PointTarget>
    OptimizationStatus Registration<NPARAM, PointSource, PointTarget>::minimize(Eigen::Matrix4f &x0_matrix)
    {

        if (l_dataset->source == nullptr || l_dataset->target == nullptr || l_dataset->tgt_search_method == nullptr)
        {
            throw std::runtime_error("Invalid dataset. Check if dataset pointers are allocated.\n");
        }

        m_final_transform = x0_matrix;
        OptimizationStatus status = registration_loop();
        return status;
    }

    template <int NPARAM, class PointSource, typename PointTarget>
    OptimizationStatus Registration<NPARAM, PointSource, PointTarget>::minimize(VectorN &x0)
    {
        so3::param2Matrix<float>(x0, m_final_transform);
        OptimizationStatus status = registration_loop();
        return status;
    }

    template <int NPARAM, typename PointSource, typename PointTarget>
    Eigen::Matrix4f Registration<NPARAM, PointSource, PointTarget>::getFinalTransformation() const
    {
        return m_final_transform;
    }

    template <int NPARAM, typename PointSource, typename PointTarget>
    void Registration<NPARAM, PointSource, PointTarget>::update_correspondences()
    {
        m_correspondences->clear();
        m_correspondences->reserve(m_source_transformed->size());

        pcl::Indices indices(m_k_neighboors);
        std::vector<float> k_distances(m_k_neighboors);

        // compute correspondences
        for (int i = 0; i < m_source_transformed->size(); ++i)
        {

            const PointSource &pt_warped = m_source_transformed->points[i];

            l_dataset->tgt_search_method->nearestKSearchT(pt_warped, m_k_neighboors, indices, k_distances);

            if (k_distances[0] > m_max_corr_dist * m_max_corr_dist)
                continue;

            // Assume normals are computed

            pcl::Correspondence correspondence;
            correspondence.index_match = indices[0];
            correspondence.index_query = i;
            m_correspondences->push_back(correspondence);
        }

        DUNA_DEBUG("source pts : %ld, target pts : %ld, corr pts: %ld\n", l_dataset->source->size(), l_dataset->target->size(), m_correspondences->size());

        if (m_correspondences->size() == 0)
        {
            throw std::runtime_error("no more correspondences.");
        }
    }

    template <int NPARAM, typename PointSource, typename PointTarget>
    OptimizationStatus Registration<NPARAM, PointSource, PointTarget>::registration_loop()
    {

        pcl::transformPointCloud(*l_dataset->source, *m_source_transformed, m_final_transform);
        // Perform optimization @ every ICP iteration
        VectorN x0_reg;
        for (int i = 0; i < m_icp_iterations; ++i)
        {
            DUNA_DEBUG_STREAM("## ICP Iteration: " << i + 1 << "/" << m_icp_iterations << " ##\n");
            update_correspondences();

            // We reset our states after we transform the cloud closer to minimum local
            x0_reg.setZero();
            OptimizationStatus status = GenericOptimizator<NPARAM>::minimize(x0_reg);

            Eigen::Matrix4f delta_transform;
            so3::param2Matrix<float>(x0_reg, delta_transform);

            pcl::transformPointCloud(*m_source_transformed, *m_source_transformed, delta_transform);

            // Increment final solution
            m_final_transform = delta_transform * m_final_transform;

            if (status == OptimizationStatus::SMALL_DELTA)
                return status;
        }

        return OptimizationStatus::MAX_IT_REACHED;
    }

}
