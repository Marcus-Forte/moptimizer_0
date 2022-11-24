#pragma once

#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_estimation.h>
#include <duna/optimizer.h>
#include <duna/logger.h>

namespace duna
{
    /* This class is a thin wrapper around the optimization object specific for point cloud registration / scan matching */
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class ScanMatchingBase
    {
    public:
        using Ptr = std::shared_ptr<ScanMatchingBase<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const ScanMatchingBase<PointSource, PointTarget, Scalar>>;

        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        // using SearchTree = pcl::search::Search<PointTarget>;
        using SearchTree = pcl::search::KdTree<PointTarget>;
        using SearchTreePtr = typename SearchTree::Ptr;

        using OptimizerPtr = typename duna::Optimizer<Scalar>::Ptr;

        using CorrespondenceEstimator = pcl::registration::CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>;
        using CorrespondenceEstimatorPtr = typename CorrespondenceEstimator::Ptr;

        using Matrix4 = typename Eigen::Matrix<Scalar, 4, 4>;

        ScanMatchingBase() : logger_("Matcher"),
                             corr_estimator_(new pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>),
                             max_corr_distance_(std::numeric_limits<float>::max()),
                             max_num_iterations_(5),
                             max_num_opt_iterations_(3)
        {
        }
        virtual ~ScanMatchingBase() = default;

        inline void setInputSource(const PointCloudSourceConstPtr &source)
        {
            source_ = source;
        }

        inline void setInputTarget(const PointCloudTargetConstPtr &target)
        {
            target_ = target;
        }

        inline void setTargetSearchTree(SearchTreePtr tree)
        {
            target_tree_ = tree;
        }

        inline PointCloudSourceConstPtr getInputSource() const
        {
            return source_;
        }

        inline PointCloudTargetConstPtr getInputTarget() const
        {
            return target_;
        }

        inline void setMaxNumIterations(int max_num_iterations)
        {
            max_num_iterations_ = max_num_iterations;
        }

        inline void setMaxNumOptIterations(int max_num_opt_iterations)
        {
            max_num_opt_iterations_ = max_num_opt_iterations;
        }

        inline int getMaxNumIterations() const
        {
            return max_num_iterations_;
        }

        inline void setMaxCorrDistance(float max_corr_dist)
        {
            max_corr_distance_ = max_corr_dist;
        }

        inline float getMaxCorrDistance() const
        {
            return max_corr_distance_;
        }

        virtual void match(Scalar *x0) = 0;

        inline logger &getLogger()
        {
            return logger_;
        }

        inline Matrix4 getFinalTransform() const
        {
            return final_transform_;
        }

    protected:
        virtual void updateCorrespondences(pcl::CorrespondencesPtr correspondences);

    protected:
        PointCloudSourceConstPtr source_;
        PointCloudTargetConstPtr target_;
        SearchTreePtr target_tree_;
        OptimizerPtr optimizer_;
        CorrespondenceEstimatorPtr corr_estimator_;
        logger logger_;
        int max_num_iterations_;
        int max_num_opt_iterations_;
        float max_corr_distance_;
        Matrix4 final_transform_;
    };
} // namespace