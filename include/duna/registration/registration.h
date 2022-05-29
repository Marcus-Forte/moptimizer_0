#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <duna/registration/registration_base.h>
#include <duna/cost_function_numerical.h>
#include <duna/cost_function_analytical.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <duna/logging.h>

#include <duna/registration/models/point2point.h>
#include <duna/registration/models/point2plane.h>

#include <unordered_set>
#include <unordered_map>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class Registration : public RegistrationBase<PointSource, PointTarget, Scalar>
    {
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

    public:
        Registration()
        {
            m_optimizer = new duna::LevenbergMarquadt<Scalar, 6>;
            m_optimizer->setMaximumIterations(3);
        }

        inline void setPoint2Plane() { m_normal_distance_mode = true; }
        inline void setPoint2Point() { m_normal_distance_mode = false; }
        void align() override;
        void align(const Matrix4 &guess) override;

    protected:
        virtual void updateCorrespondences() override;
        virtual void registrationLoop();
        bool m_normal_distance_mode = false;
        std::unique_ptr<std::unordered_map<int, pcl::Normal>> m_normal_map;
    };
}

#endif