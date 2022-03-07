#pragma once

#include "duna/registration.h"

template <int NPARAM>
Registration<NPARAM>::Registration(CostFunction<NPARAM> *cost) : GenericOptimizator<NPARAM>(cost)
{
    m_source_transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    m_correspondences = pcl::make_shared<pcl::Correspondences>();

    RegistrationCost<NPARAM> *l_cost = reinterpret_cast<RegistrationCost<NPARAM> *>(cost);
    l_cost->setCorrespondencesPtr(m_correspondences);
    l_cost->setTransformedSourcePtr(m_source_transformed);
    l_dataset = reinterpret_cast<reg_cost_data_t *>(cost->getDataset());

    // By Default, we want to set the internal optimization loop to a single iteration to allow ICP to transform source more often.
    Optimizator<NPARAM>::setMaxOptimizationIterations(1);
}

template <int NPARAM>
Registration<NPARAM>::~Registration()
{
}

template <int NPARAM>
opt_status Registration<NPARAM>::minimize(VectorN &x0)
{

    m_final_transform = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f init_transform;
    so3::param2Matrix<float>(x0, init_transform);

    pcl::transformPointCloud(*l_dataset->source, *m_source_transformed, init_transform);

    // Perform optimization @ every ICP iteration
    for (int i = 0; i < m_icp_iterations; ++i)
    {
        DUNA_DEBUG_STREAM("## ICP Iteration: " << i + 1 << "/" << m_icp_iterations << " ##\n");
        update_correspondences();

        opt_status status = GenericOptimizator<NPARAM>::minimize(x0);

        Eigen::Matrix4f delta_transform;
        so3::param2Matrix<float>(x0, delta_transform);

        pcl::transformPointCloud(*m_source_transformed, *m_source_transformed, delta_transform);
        x0.setZero();

        m_final_transform = delta_transform * m_final_transform;

        if (status == opt_status::SMALL_DELTA)
            return status;
    }

    return opt_status::MAX_IT_REACHED;
}

template <int NPARAM>
Eigen::Matrix4f Registration<NPARAM>::getFinalTransformation() const
{
    return m_final_transform;
}

template <int NPARAM>
void Registration<NPARAM>::update_correspondences()
{
    m_correspondences->clear();
    m_correspondences->reserve(m_source_transformed->size());

    pcl::Indices indices(m_k_neighboors);
    std::vector<float> k_distances(m_k_neighboors);

    // compute correspondences
    for (int i = 0; i < m_source_transformed->size(); ++i)
    {

        const pcl::PointXYZ &pt_warped = m_source_transformed->points[i];

        l_dataset->tgt_kdtree->nearestKSearch(pt_warped, m_k_neighboors, indices, k_distances);

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
