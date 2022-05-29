#ifndef REGISTRATION3DOF_H
#define REGISTRATION3DOF_H
#include <duna/registration/registration.h>
#include <duna/registration/models/point2plane3dof.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar>
    class Registration3DOF : public Registration<PointSource, PointTarget, Scalar>
    {
    public:
        Registration3DOF()
        {
            // TODO remove from parent constructor
            m_optimizer = new LevenbergMarquadt<Scalar, 3>;
            m_optimizer->setMaximumIterations(3);
        }

    protected:
        using Matrix4 = typename RegistrationBase<PointSource, PointTarget, Scalar>::Matrix4;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_max_icp_iterations;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_correspondences;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_transformed_source;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_source;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_target;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_max_corr_dist;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_target_search_method;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_nearest_k;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_final_transformation;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_optimizer;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_current_iterations;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_optimizator_status;

        using PointCloudNormalT = pcl::PointCloud<pcl::Normal>;
        using Registration<PointSource, PointTarget, Scalar>::m_normal_distance_mode;
        using Registration<PointSource, PointTarget, Scalar>::m_normal_map;

        void registrationLoop() override;
    };
}
#endif