#ifndef REGISTRATION_BASE_H
#define REGISTRATION_BASE_H

#include <duna/optimizer.h>

#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/search.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class RegistrationBase
    {
    public:
        using PointCloudSourcePtr = typename pcl::PointCloud<PointSource>::Ptr;
        using PointCloudTargetPtr = typename pcl::PointCloud<PointTarget>::Ptr;
        using PointCloudSourceConstPtr = typename pcl::PointCloud<PointSource>::ConstPtr;
        using PointCloudTargetConstPtr = typename pcl::PointCloud<PointTarget>::ConstPtr;

        using TargetPointCloudSearch = typename pcl::search::Search<PointTarget>;
        using TargetPointCloudSearchPtr = typename pcl::search::Search<PointTarget>::Ptr;
        using TargetPointCloudSearchConstPtr = typename pcl::search::Search<PointTarget>::ConstPtr;

        using Matrix4f = Eigen::Matrix4f;

        RegistrationBase()
        {
            m_transformed_source.reset(new pcl::PointCloud<PointSource>);
        }
        virtual ~RegistrationBase() = default;

        void setMaximumICPIterations(const unsigned int max_it) { m_maximum_icp_iterations = max_it; }
        void setMaximumCorrespondenceDistance(const float max_corr_dist) { m_maximum_correspondences_distance = max_corr_dist; }
        void setSourceCloud(PointCloudSourceConstPtr source) { m_source = source; }
        void setTargetCloud(PointCloudTargetConstPtr target) { m_target = target; }
        void setTargetSearchMethod(TargetPointCloudSearchConstPtr target_kdtree) {m_target_kdtree =  target_kdtree; }

        Eigen::Matrix4f getFinalTransformation()
        {
            return m_final_transformation;
        }

        OptimizationStatus getOptimizationStatus() const 
        {
            return m_optimization_status;
        }
        virtual void align() = 0;
        virtual void align(const Matrix4f& guess) = 0;

    protected:
        // TODO make it more generic
        Optimizer<Scalar, 6, 1> *m_optimizer;

        PointCloudTargetConstPtr m_target;
        PointCloudSourceConstPtr m_source;
        TargetPointCloudSearchConstPtr m_target_kdtree;
        PointCloudSourcePtr m_transformed_source;
        pcl::Correspondences m_correspondences;

        unsigned int m_maximum_icp_iterations;
        float m_maximum_correspondences_distance;
        unsigned int m_nearest_k = 5;

        // Alignment transformation
        Eigen::Matrix4f m_final_transformation;

        OptimizationStatus m_optimization_status;

    };
}
#endif