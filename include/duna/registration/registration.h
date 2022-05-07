#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <duna/registration/registration_base.h>
#include <duna/levenberg_marquadt.h>
#include <duna/registration/registration_model.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class Registration : public RegistrationBase<PointSource,PointTarget,Scalar>
    {
        public:
        using PointCloudSourcePtr = typename RegistrationBase<PointSource,PointTarget,Scalar>::PointCloudSourcePtr;
        using PointCloudTargetPtr = typename RegistrationBase<PointSource,PointTarget,Scalar>::PointCloudTargetPtr;
        using PointCloudSourceConstPtr = typename RegistrationBase<PointSource,PointTarget,Scalar>::PointCloudSourceConstPtr;
        using PointCloudTargetConstPtr = typename RegistrationBase<PointSource,PointTarget,Scalar>::PointCloudTargetConstPtr;
        using Matrix4 = typename RegistrationBase<PointSource,PointTarget,Scalar>::Matrix4;

        Registration();
        virtual ~Registration() = default;

        void align() override;       
        void align(const Matrix4& guess);

        protected:
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_maximum_icp_iterations;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_maximum_correspondences_distance;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_final_transformation;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_transformed_source;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_correspondences;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_target;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_source;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_optimizer;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_optimization_status;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_nearest_k;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_target_kdtree;
        

        void update_correspondences();
        void registration_loop();

    };
}
#endif