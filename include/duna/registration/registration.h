#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <duna/registration/registration_base.h>
#include <duna/levenberg_marquadt.h>
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
        Registration()
        {
            optimizer = new LevenbergMarquadt<Scalar,6,1>;
        }

        // Implement ICP
        void align() override
        {

        }

        protected:
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_maximum_icp_iterations;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_maximum_correspondences_distance;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_target;
        using RegistrationBase<PointSource,PointTarget,Scalar>::m_source;
        using RegistrationBase<PointSource,PointTarget,Scalar>::optimizer;

    };
}
#endif