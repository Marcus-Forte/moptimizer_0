#include <duna/registration/registration_3dof.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    void Registration3DOF<PointSource, PointTarget, Scalar>::registrationLoop()
    {
        pcl::transformPointCloud(*m_source, *m_transformed_source, m_final_transformation);

        CostFunctionBase<Scalar> *cost;

        // TODO abstract
        // if (m_normal_distance_mode)
        {
            cost = new duna::CostFunctionNumericalDiff<Point2Plane3DOF<PointSource, PointTarget, Scalar>, Scalar, 3, 1>(
                new Point2Plane3DOF<PointSource, PointTarget, Scalar>(*m_transformed_source, *m_target, *m_normal_map, m_correspondences));
            std::cerr << "point2plane\n";
        }
        // else
        // {
        //     cost = new duna::CostFunctionNumericalDiff<Point2Plane3DOF<PointSource, PointTarget, Scalar>, Scalar, 3, 1>(
        //         new Point2Plane3DOF<PointSource, PointTarget, Scalar>(*m_transformed_source, *m_target, m_correspondences));
        //     std::cerr << "point2point\n";
        // }

        m_optimizer->setCost(cost);

        // TODO abstract
        Eigen::Matrix<Scalar, 3, 1> x0;

        Matrix4 delta_transform;

        for (m_current_iterations = 0; m_current_iterations < m_max_icp_iterations; ++m_current_iterations)
        {
            DUNA_DEBUG("ICP ITERATION #%d / %d \n", m_current_iterations + 1, m_max_icp_iterations);
            Registration<PointSource,PointTarget,Scalar>::updateCorrespondences();

            cost->setNumResiduals(m_correspondences.size());

            x0.setZero();
            OptimizationStatus status = m_optimizer->minimize(x0.data());

            if (status == OptimizationStatus::NUMERIC_ERROR)
                throw std::runtime_error("Optimizer Numeric error");

            // TODO abstract
            so3::convert3DOFParameterToMatrix(x0.data(), delta_transform);

            pcl::transformPointCloud(*m_transformed_source, *m_transformed_source, delta_transform);

            m_final_transformation = delta_transform * m_final_transformation;

            if (status == OptimizationStatus::SMALL_DELTA || status == OptimizationStatus::CONVERGED)
                return;
        }

        delete cost;
    }

    template class Registration3DOF<pcl::PointXYZ, pcl::PointXYZ, double>;
    template class Registration3DOF<pcl::PointXYZ, pcl::PointXYZ, float>;

    template class Registration3DOF<pcl::PointNormal, pcl::PointNormal, double>;
    template class Registration3DOF<pcl::PointNormal, pcl::PointNormal, float>;

} // namespace