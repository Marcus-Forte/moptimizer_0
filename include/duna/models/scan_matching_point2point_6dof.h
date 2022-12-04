#pragma once

#include <duna/models/scan_matching_base.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    class ScanMatching6DOFPoint2Point : public ScanMatchingBase<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = std::shared_ptr<ScanMatching6DOFPoint2Point>;
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using KdTree = pcl::search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;

        ScanMatching6DOFPoint2Point(PointCloudSourceConstPtr source,
                                    PointCloudTargetConstPtr target,
                                    KdTreePtr kdtree_target) : ScanMatchingBase<PointSource, PointTarget, Scalar>(source, target, kdtree_target)
        {}

        virtual ~ScanMatching6DOFPoint2Point() = default;

        void setup(const Scalar *x) override
        {
            so3::convert6DOFParameterToMatrix(x, transform_);
        }

        bool f(const Scalar *x, Scalar *f_x, unsigned int index) override
        {
            if (index >= correspondences_.size())
                return false;

            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(src_pt.x, src_pt.y, src_pt.z, 1);
            Eigen::Matrix<Scalar, 4, 1> tgt_(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);

            Eigen::Matrix<Scalar, 4, 1> warped_src_ = transform_ * src_;
            warped_src_[3] = 0;

            Eigen::Matrix<Scalar, 4, 1> error = warped_src_ - tgt_;

            // Much faster than norm.
            f_x[0] = error[0];
            f_x[1] = error[1];
            f_x[2] = error[2];
            return true;
        }

    protected:
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::source_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::target_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::kdtree_target_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::transformed_source_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::correspondences_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::transform_;
    };
}