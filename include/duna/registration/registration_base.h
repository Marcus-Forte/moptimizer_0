#ifndef REGISTRATIONBASE_H
#define REGISTRATIONBASE_H

#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/correspondence.h>

#include <duna/levenberg_marquadt.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class RegistrationBase
    {
    public:
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using TargetSearchMethod = pcl::search::Search<PointTarget>;
        using TargetSearchMethodPtr = typename TargetSearchMethod::Ptr;
        using TargetSearchMethodConstPtr = typename TargetSearchMethod::ConstPtr;

        using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;

    public:
        RegistrationBase()
        {
            m_transformed_source.reset(new PointCloudSource);
            m_optimizer = new duna::LevenbergMarquadt<Scalar,6,1>;

            // Run only one optimization loop, then transform source closer to target.
            m_optimizer->setMaximumIterations(1);
            
        }

        void setInputSource(PointCloudSourceConstPtr source) { m_source = source; }
        void setInputTarget(PointCloudTargetConstPtr target) { m_target = target; }
        void setMaximumICPIterations(unsigned int max_iterations) { m_max_icp_iterations = max_iterations; }
        void setMaximumOptimizerIterations(unsigned int max_iterations ) { m_optimizer->setMaximumIterations(max_iterations); }
        void setTargetSearchMethod(TargetSearchMethodConstPtr method) { m_target_search_method = method; }
        void setMaximumCorrespondenceDistance( float max_distance ) { m_max_corr_dist = max_distance; }
        void setMaxNearestK(unsigned int nearest_k) { m_nearest_k = nearest_k;}

        Matrix4 getFinalTransformation() { return m_final_transformation; }

        virtual void align() = 0;
        virtual void align(const Matrix4 &guess) = 0;

    protected:

        virtual void updateCorrespondences() = 0;
        
        unsigned int m_max_icp_iterations = 10;
        PointCloudSourceConstPtr m_source;
        PointCloudTargetConstPtr m_target;

        TargetSearchMethodConstPtr m_target_search_method;
        float m_max_corr_dist = 1;
        unsigned int m_nearest_k = 5;

        PointCloudSourcePtr m_transformed_source;

        Matrix4 m_final_transformation;
        pcl::Correspondences m_correspondences;
        Optimizer<Scalar, 6, 1> *m_optimizer;
    };
}

#endif