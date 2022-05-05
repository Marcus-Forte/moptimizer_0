#ifndef REGISTRATION_BASE_H
#define REGISTRATION_BASE_H

#include <duna/optimizer.h>
#include <pcl/point_cloud.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class RegistrationBase
    {
    public:
        using PointCloudSourcePtr = typename pcl::PointCloud<PointSource>::Ptr;
        using PointCloudTargetPtr = typename pcl::PointCloud<PointTarget>::Ptr;
        using PointCloudSourceConstPtr = typename pcl::PointCloud<PointSource>::ConstPtr;
        using PointCloudTargetConstPtr = typename pcl::PointCloud<PointTarget>::ConstPtr;

        RegistrationBase()
        {
            m_final_transformation = Eigen::Matrix4f::Identity();
        }
        virtual ~RegistrationBase() = default;

        void setMaximumICPIterations(const unsigned int max_it) { m_maximum_icp_iterations = max_it; }
        void setMaximumCorrespondenceDistance(const float max_corr_dist) { m_maximum_correspondences_distance = max_corr_dist; }
        void setSourceCloud(PointCloudSourceConstPtr source) { m_source = source; }
        void setTargetCloud(PointCloudTargetConstPtr target) { m_target = target; }

        Eigen::Matrix4f getFinalTransformation()
        {
            return m_final_transformation;
        }
        virtual void align() = 0;

    protected:
        Optimizer<Scalar, 6, 1> *optimizer;
        
        PointCloudTargetConstPtr m_target;
        PointCloudSourceConstPtr m_source;
        unsigned int m_maximum_icp_iterations;
        float m_maximum_correspondences_distance;

        Eigen::Matrix4f m_final_transformation;
    };
}
#endif