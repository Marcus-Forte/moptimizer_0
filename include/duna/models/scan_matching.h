#pragma once

#include <duna/models/scan_matching_base.h>
#include <cmath>

namespace duna
{
    /* Unified point to plane 3DOF registration model. */
    template <typename PointSource, typename PointTarget, typename Scalar>
    class ScanMatching3DOFPoint2Plane : public ScanMatchingBase<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = std::shared_ptr<ScanMatching3DOFPoint2Plane>;
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using KdTree = pcl::search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;

        ScanMatching3DOFPoint2Plane(PointCloudSourceConstPtr source,
                                    PointCloudTargetConstPtr target,
                                    KdTreePtr kdtree_target) : ScanMatchingBase<PointSource, PointTarget, Scalar>(source, target, kdtree_target)
        {
        }

        virtual ~ScanMatching3DOFPoint2Plane() = default;

        void setup(const Scalar *x) override
        {
            so3::convert3DOFParameterToMatrix(x, transform_);
        }

        virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override
        {
            if (index >= correspondences_.size())
                return false;

            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            if (std::isnan(tgt_pt.normal_x))
            {
                return false;
            }

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_normal_(static_cast<Scalar>(tgt_pt.normal_x), static_cast<Scalar>(tgt_pt.normal_y), static_cast<Scalar>(tgt_pt.normal_z), 0.0);

            Eigen::Matrix<Scalar, 4, 1> &&warped_src_ = transform_ * src_;

            f_x[0] = (warped_src_ - tgt_).dot(tgt_normal_);

            return true;
        }

        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) override
        {
            if (index >= correspondences_.size())
                return false;

            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            if (std::isnan(tgt_pt.normal_x))
            {
                return false;
            }

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_normal_(static_cast<Scalar>(tgt_pt.normal_x), static_cast<Scalar>(tgt_pt.normal_y), static_cast<Scalar>(tgt_pt.normal_z), 0.0);

            Eigen::Matrix<Scalar, 4, 1> &&warped_src_ = transform_ * src_;

            f_x[0] = (warped_src_ - tgt_).dot(tgt_normal_);
            jacobian[0] = tgt_normal_[2] * src_[1] - tgt_normal_[1] * src_[2];
            jacobian[1] = tgt_normal_[0] * src_[2] - tgt_normal_[2] * src_[0];
            jacobian[2] = tgt_normal_[1] * src_[0] - tgt_normal_[0] * src_[1];

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

    template <typename PointSource, typename PointTarget, typename Scalar>
    class ScanMatching3DOFPoint2Point : public ScanMatchingBase<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = std::shared_ptr<ScanMatching3DOFPoint2Point>;
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using KdTree = pcl::search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;

        using JacobianType = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

        ScanMatching3DOFPoint2Point(PointCloudSourceConstPtr source,
                                    PointCloudTargetConstPtr target,
                                    KdTreePtr kdtree_target) : ScanMatchingBase<PointSource, PointTarget, Scalar>(source, target, kdtree_target)
        {
        }

        virtual ~ScanMatching3DOFPoint2Point() = default;

        void setup(const Scalar *x) override
        {
            so3::convert3DOFParameterToMatrix(x, transform_);
        }

        virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override
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

        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) override
        {
            if (index >= correspondences_.size())
                return false;

            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);

            Eigen::Matrix<Scalar, 4, 1> warped_src_ = transform_ * src_;
            warped_src_[3] = 0;

            Eigen::Matrix<Scalar, 4, 1> error = warped_src_ - tgt_;

            f_x[0] = error[0];
            f_x[1] = error[1];
            f_x[2] = error[2];

            Eigen::Map<JacobianType> jacobian_map(jacobian);
            Eigen::Matrix<Scalar, 3, 3> skew;
            skew << SKEW_SYMMETRIC_FROM(src_);
            jacobian_map = -1.0 * skew;

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

    /* Unified point to point 6DOF registration model. */
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

        using JacobianType = Eigen::Matrix<Scalar, 3, 6, Eigen::RowMajor>;

        ScanMatching6DOFPoint2Point(PointCloudSourceConstPtr source,
                                    PointCloudTargetConstPtr target,
                                    KdTreePtr kdtree_target) : ScanMatchingBase<PointSource, PointTarget, Scalar>(source, target, kdtree_target)
        {
        }

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

        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) override
        {
            if (index >= correspondences_.size())
                return false;

            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);

            Eigen::Matrix<Scalar, 4, 1> warped_src_ = transform_ * src_;
            warped_src_[3] = 0;

            Eigen::Matrix<Scalar, 4, 1> error = warped_src_ - tgt_;

            f_x[0] = error[0];
            f_x[1] = error[1];
            f_x[2] = error[2];

            Eigen::Map<JacobianType> jacobian_map(jacobian);
            Eigen::Matrix<Scalar, 3, 3> skew;
            skew << SKEW_SYMMETRIC_FROM(src_);
            jacobian_map.template block<3, 3>(0, 0) = Eigen::Matrix<Scalar, 3, 3>::Identity();
            jacobian_map.template block<3, 3>(0, 3) = -1.0 * skew;

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

    /* Unified point to plane 6DOF registration model. */
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
        {
        }

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
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::source_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::target_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::kdtree_target_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::transformed_source_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::correspondences_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::transform_;
    };
}