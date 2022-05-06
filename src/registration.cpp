#include <duna/registration/registration.h>
#include <duna/registration/registration_model.h>

#include <pcl/common/transforms.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    Registration<PointSource, PointTarget, Scalar>::Registration()
    {
        // 6DOF
        m_optimizer = new LevenbergMarquadt<Scalar, 6, 1>;
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

        for (int it = 0; it < m_maximum_icp_iterations; ++it)
        {
            DUNA_DEBUG_STREAM("## ICP Iteration: " << it + 1 << "/" << m_maximum_icp_iterations << " ##\n");
            update_correspondences();
            // TODO This looks HORRIBLE
            m_optimizer->setCost(new duna::CostFunction<float, 6, 1>(
                new RegistrationModel<PointSource, PointTarget>(*m_transformed_source, *m_target, m_correspondences), m_correspondences.size()));

            x0.setZero();
            m_optimizer->minimize(x0);

            std::cout << x0;

            // Minimize
            // Call optimizator
        }
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