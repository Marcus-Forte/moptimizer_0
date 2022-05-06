#include <duna/registration/registration.h>
#include <duna/registration/registration_model.h>

#include <pcl/common/transforms.h>
#include <duna/stopwatch.hpp>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    Registration<PointSource, PointTarget, Scalar>::Registration()
    {
        // 6DOF
        m_optimizer = new LevenbergMarquadt<Scalar, 6, 1>;
        m_optimizer->setMaximumIterations(1);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::align()
    {
        m_final_transformation.setIdentity();

        registration_loop();
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::align(const Matrix4f &guess)
    {
        m_final_transformation = guess;

        registration_loop();
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::registration_loop()
    {

        pcl::transformPointCloud(*m_source, *m_transformed_source, m_final_transformation);
        Eigen::Matrix<Scalar, 6, 1> x0;

        // TODO This looks HORRIBLE
        utilities::Stopwatch stopwatch(true);

        auto *cost = new duna::CostFunction<float, 6, 1>(new RegistrationModel<PointSource, PointTarget>(*m_transformed_source, *m_target, m_correspondences));

        utilities::Stopwatch stopwatch_total(true);

        m_optimizer->setCost(cost);
        stopwatch_total.tick();

        for (int it = 0; it < m_maximum_icp_iterations; ++it)
        {
            DUNA_DEBUG_STREAM("## ICP Iteration: " << it + 1 << "/" << m_maximum_icp_iterations << " ##\n");
            update_correspondences();
            cost->setNumResiduals(m_correspondences.size());

            x0.setZero();

            stopwatch.tick();
            m_optimization_status = m_optimizer->minimize(x0);
            stopwatch.tock("Minimize");

            if (m_optimization_status == OptimizationStatus::SMALL_DELTA)
            {
                delete cost;
                stopwatch_total.tock("registration loop");

                return;
            }

            Eigen::Matrix4f delta_transform;
            so3::convert6DOFParameterToMatrix(x0.data(), delta_transform);

            pcl::transformPointCloud(*m_transformed_source, *m_transformed_source, delta_transform);

            m_final_transformation = delta_transform * m_final_transformation;
        }

        delete cost;
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration<PointSource, PointTarget, Scalar>::update_correspondences()
    {
        m_correspondences.clear();
        m_correspondences.reserve(m_transformed_source->size());

        std::vector<int> indices(m_nearest_k);
        std::vector<float> sqrd_distances(m_nearest_k);

        for (int i = 0; i < m_transformed_source->size(); ++i)
        {
            const PointSource &point_warped = m_transformed_source->points[i];

            m_target_kdtree->nearestKSearchT(point_warped, m_nearest_k, indices, sqrd_distances);

            if (sqrd_distances[0] > m_maximum_correspondences_distance * m_maximum_correspondences_distance)
                continue;

            pcl::Correspondence correspondence;
            correspondence.index_match = indices[0];
            correspondence.index_query = i;
            m_correspondences.push_back(correspondence);
        }

        if (m_correspondences.size() == 0)
        {
            throw std::runtime_error("no more correspondences.");
        }
    }

    // Instantiations
    template class Registration<pcl::PointXYZ, pcl::PointXYZ, float>;
}