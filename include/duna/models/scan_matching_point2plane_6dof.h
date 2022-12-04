#pragma once

#include <duna/models/scan_matching_base.h>

/* Unified point to plane 3DOF registration model. */
namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    class ScanMatching6DOFPoint2Plane : public ScanMatchingBase<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = std::shared_ptr<ScanMatching6DOFPoint2Plane>;
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using KdTree = pcl::search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;

        ScanMatching6DOFPoint2Plane(PointCloudSourceConstPtr source,
                         PointCloudTargetConstPtr target,
                         KdTreePtr kdtree_target) : ScanMatchingBase<PointSource, PointTarget, Scalar>(source, target, kdtree_target)
        {}

        virtual ~ScanMatching6DOFPoint2Plane() = default;

        void setup(const Scalar *x) override
        {
            so3::convert6DOFParameterToMatrix(x, transform_);
        }
    
        virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override
        {
            if (index >= correspondences_.size())
                return false;

            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_normal_(static_cast<Scalar>(tgt_pt.normal_x), static_cast<Scalar>(tgt_pt.normal_y), static_cast<Scalar>(tgt_pt.normal_z), 0.0);

            Eigen::Matrix<Scalar, 4, 1> &&warped_src_ = transform_ * src_;

            f_x[0] = (warped_src_ - tgt_).dot(tgt_normal_);
            return true;
        }
    protected:
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::source_;
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::target_;
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::kdtree_target_;
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::transformed_source_;
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::correspondences_;
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::transform_;
    };
}