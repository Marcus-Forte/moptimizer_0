#include <duna/registration/registration.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::align()
    {
        m_final_transformation.setIdentity();
        align(m_final_transformation);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::align(const Matrix4 &guess)
    {
        if (!m_target_search_method)
            throw std::runtime_error("No KD found");

        if (!m_target)
            throw std::runtime_error("No Target Point Cloud found");

        if (!m_source)
            throw std::runtime_error("No Source Point Cloud found");

        DUNA_DEBUG("Target pts: %ld \n", m_target->size());
        DUNA_DEBUG("Source pts: %ld \n", m_source->size());

        // Prepare data
        if (m_normal_distance_mode)
        {
            target_normals.reset(new PointCloudNormalT);
            target_normals->resize(m_target->size());
        }

        registrationLoop();
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::updateCorrespondences()
    {
        m_correspondences.clear();

        std::vector<float> sqrd_distances(m_nearest_k);
        std::vector<int> indices(m_nearest_k);

        float max_sqrd_dist = m_max_corr_dist * m_max_corr_dist;
        if (m_normal_distance_mode)
        {
            for (int i = 0; i < m_transformed_source->size(); ++i)
            {
                const PointSource &queryPoint = m_transformed_source->points[i];

                m_target_search_method->nearestKSearchT(queryPoint, m_nearest_k, indices, sqrd_distances);

                if (sqrd_distances[0] > max_sqrd_dist)
                    continue;

                // TODO precompute normals
                pcl::Correspondence corr;
                corr.index_query = i;
                corr.index_match = indices[0];
                m_correspondences.push_back(corr);

                // TODO avoid recompute
                Eigen::Vector4f normal_;
                float unused;
                pcl::computePointNormal(*m_target, indices, normal_, unused);

                target_normals->points[indices[0]].normal_x = normal_[0];
                target_normals->points[indices[0]].normal_y = normal_[1];
                target_normals->points[indices[0]].normal_z = normal_[2];
            }
        }
        else
        {
            for (int i = 0; i < m_transformed_source->size(); ++i)
            {
                const PointSource &queryPoint = m_transformed_source->points[i];

                m_target_search_method->nearestKSearchT(queryPoint, m_nearest_k, indices, sqrd_distances);

                if (sqrd_distances[0] > max_sqrd_dist)
                    continue;

                // TODO precompute normals
                pcl::Correspondence corr;
                corr.index_query = i;
                corr.index_match = indices[0];
                m_correspondences.push_back(corr);
            }
        }

        if (m_correspondences.size() == 0)
            throw std::runtime_error("No correspondences found.");
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::registrationLoop()
    {
        pcl::transformPointCloud(*m_source, *m_transformed_source, m_final_transformation);

        CostFunctionBase<Scalar, 6, 1> *cost;

        if (m_normal_distance_mode)
        {
            cost = new duna::CostFunctionNumericalDiff<Point2Plane<PointSource, PointTarget, Scalar>, Scalar, 6, 1>(
                new Point2Plane<PointSource, PointTarget, Scalar>(*m_transformed_source, *m_target, *target_normals, m_correspondences)
                );
            std::cerr << "point2plane\n";
        }
        else
        {
            cost = new duna::CostFunctionNumericalDiff<Point2Point<PointSource, PointTarget, Scalar>, Scalar, 6, 1>(
                new Point2Point<PointSource, PointTarget, Scalar>(*m_transformed_source, *m_target, m_correspondences)
                );
            std::cerr << "point2point\n";
        }

        m_optimizer->setCost(cost);

        Eigen::Matrix<Scalar, 6, 1> x0;

        Matrix4 delta_transform;

        for (m_current_iterations = 0; m_current_iterations < m_max_icp_iterations; ++m_current_iterations)
        {
            DUNA_DEBUG("ICP ITERATION #%d / %d \n", m_current_iterations + 1, m_max_icp_iterations);
            updateCorrespondences();

            cost->setNumResiduals(m_correspondences.size());

            x0.setZero();
            OptimizationStatus status = m_optimizer->minimize(x0.data());

            if (status == OptimizationStatus::NUMERIC_ERROR)
                throw std::runtime_error("Optimizer Numeric error");

            so3::convert6DOFParameterToMatrix(x0.data(), delta_transform);

            pcl::transformPointCloud(*m_transformed_source, *m_transformed_source, delta_transform);

            m_final_transformation = delta_transform * m_final_transformation;

            if (status == OptimizationStatus::SMALL_DELTA || status == OptimizationStatus::CONVERGED)
                return;
        }

        delete cost;
    }

    template class Registration<pcl::PointXYZ, pcl::PointXYZ, double>;
    template class Registration<pcl::PointXYZ, pcl::PointXYZ, float>;

    template class Registration<pcl::PointNormal, pcl::PointNormal, double>;
    template class Registration<pcl::PointNormal, pcl::PointNormal, float>;

}