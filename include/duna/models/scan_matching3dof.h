#pragma once

#include "duna/so3.h"
#include "duna/model.h"
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_estimation.h>

/* Unified point to plane 3DOF registration model. */
namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    class ScanMatching3DOF : public BaseModelJacobian<Scalar>
    {
    public:
        using Ptr = std::shared_ptr<ScanMatching3DOF>;
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using KdTree = pcl::search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;

        ScanMatching3DOF(PointCloudSourceConstPtr source,
                         PointCloudTargetConstPtr target,
                         KdTreePtr kdtree_target) : source_(source),
                                                    target_(target),
                                                    kdtree_target_(kdtree_target),
                                                    maximum_corr_dist_(std::numeric_limits<double>::max())
        {
            if (!source_ || source_->size() == 0)
                duna::logger::log_error("No points at source cloud!");

            if (!target_ || target_->size() == 0)
                duna::logger::log_error("No points at target cloud!");

            if (!kdtree_target_)
                duna::logger::log_error("No target Kdtree!");

            corr_estimator_.setInputTarget(target_);
            corr_estimator_.setSearchMethodTarget(kdtree_target_);
            transformed_source_.reset(new PointCloudSource);
        }

        virtual ~ScanMatching3DOF() = default;

        virtual void setup(const Scalar *x) override
        {
            so3::convert3DOFParameterToMatrix(x, transform_);

            s_alpha_ = std::sin(x[0]);
            c_alpha_ = std::cos(x[0]);

            s_beta_ = std::sin(x[1]);
            c_beta_ = std::cos(x[1]);

            s_gamma_ = std::sin(x[2]);
            c_gamma_ = std::cos(x[2]);
        }

        virtual void update(const Scalar *x) override
        {
            so3::convert3DOFParameterToMatrix(x, transform_);
            pcl::transformPointCloud(*source_, *transformed_source_, transform_);

            duna::logger::log_debug("Updating correspondences...");

            corr_estimator_.setInputSource(transformed_source_);
            corr_estimator_.determineCorrespondences(correspondences_, maximum_corr_dist_);

            duna::logger::log_debug("found: %d / %d", correspondences_.size(), source_->size());
        }

        bool operator()(const Scalar *x, Scalar *f_x, unsigned int index)
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

        virtual void df(const Scalar *x, Scalar *jacobian, unsigned int index)
        {
            if (index >= correspondences_.size())
                return;
            const PointSource &src_pt = source_->points[correspondences_[index].index_query];
            const PointTarget &tgt_pt = target_->points[correspondences_[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            // Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);
            Eigen::Matrix<Scalar, 3, 1> tgt_normal_(static_cast<Scalar>(tgt_pt.normal_x), static_cast<Scalar>(tgt_pt.normal_y), static_cast<Scalar>(tgt_pt.normal_z));

            // Not sure why we multiply by two. Numerical Diff comparison suggested that.
            jacobian[0] = ((s_gamma_ * s_alpha_ + c_gamma_ * s_beta_ * c_alpha_) * src_[1] + (s_gamma_ * c_alpha_ - c_gamma_ * s_beta_ * s_alpha_) * src_[2]) * tgt_normal_[0] +
                          ((-c_gamma_ * s_alpha_ + s_gamma_ * s_beta_ * c_alpha_) * src_[1] + (-c_gamma_ * c_alpha_ - s_gamma_ * s_beta_ * s_alpha_) * src_[2]) * tgt_normal_[1] +
                          ((c_beta_ * c_alpha_) * src_[1] + (-c_beta_ * s_alpha_) * src_[2]) * tgt_normal_[2];

            jacobian[1] = ((-c_gamma_ * s_beta_) * src_[0] + (c_gamma_ * c_beta_ * s_alpha_) * src_[1] + (c_gamma_ * c_beta_ * c_alpha_) * src_[2]) * tgt_normal_[0] +
                          ((-s_gamma_ * s_beta_) * src_[0] + (s_gamma_ * c_beta_ * s_alpha_) * src_[1] + (s_gamma_ * c_beta_ * c_alpha_) * src_[2]) * tgt_normal_[1] +
                          ((-c_beta_) * src_[0] + (-s_beta_ * s_alpha_) * src_[1] + (-s_beta_ * c_alpha_) * src_[2]) * tgt_normal_[2];

            jacobian[2] = ((-s_gamma_ * c_beta_) * src_[0] + (-c_gamma_ * c_alpha_ - s_gamma_ * s_beta_ * s_alpha_) * src_[1] + (c_gamma_ * s_alpha_ - s_gamma_ * s_beta_ * c_alpha_) * src_[2]) * tgt_normal_[0] +
                          ((c_gamma_ * c_beta_) * src_[0] + (-s_gamma_ * c_alpha_ + c_gamma_ * s_beta_ * s_alpha_) * src_[1] + (s_gamma_ * s_alpha_ + c_gamma_ * s_beta_ * c_alpha_) * src_[2]) * tgt_normal_[1];

            jacobian[0] /= 2;
            jacobian[1] /= 2;
            jacobian[2] /= 2;

        }

        inline void setMaximumCorrespondenceDistance(double distance)
        {
            maximum_corr_dist_ = distance;
        }

    protected:
        PointCloudSourceConstPtr source_;
        PointCloudTargetConstPtr target_;
        KdTreePtr kdtree_target_;
        PointCloudSourcePtr transformed_source_;
        pcl::Correspondences correspondences_;
        Eigen::Matrix<Scalar, 4, 4> transform_;
        pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar> corr_estimator_;

        // Angles
        Scalar s_alpha_;
        Scalar c_alpha_;

        Scalar s_beta_;
        Scalar c_beta_;

        Scalar s_gamma_;
        Scalar c_gamma_;

        // Parameters
        double maximum_corr_dist_;
    };
}