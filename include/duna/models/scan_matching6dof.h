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
    class ScanMatching6DOF : public BaseModel<Scalar>
    {
    public:
        using Ptr = std::shared_ptr<ScanMatching6DOF>;
        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using KdTree = pcl::search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;

        ScanMatching6DOF(PointCloudSourceConstPtr source,
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
            corr_estimator_.setSearchMethodTarget(kdtree_target_, true); // Never recompute.
            transformed_source_.reset(new PointCloudSource);
        }

        virtual ~ScanMatching6DOF() = default;

        virtual void setup(const Scalar *x) override
        {
            so3::convert6DOFParameterToMatrix(x, transform_);
        }

        virtual void update(const Scalar *x) override
        {
            so3::convert6DOFParameterToMatrix(x, transform_);
            pcl::transformPointCloud(*source_, *transformed_source_, transform_);

            duna::logger::log_debug("Updating correspondences... @", maximum_corr_dist_);

            corr_estimator_.setInputSource(transformed_source_);
            corr_estimator_.determineCorrespondences(correspondences_, maximum_corr_dist_);

            duna::logger::log_debug("found: %d / %d", correspondences_.size(), source_->size());

            // copy
            if (corr_rejectors.size())
            {
                pcl::CorrespondencesPtr tmp_corrs(new pcl::Correspondences(correspondences_));
                for (int i = 0; i < corr_rejectors.size(); ++i)
                {
                    duna::logger::log_debug("Using rejector: %s", corr_rejectors[i]->getClassName().c_str());
                    corr_rejectors[i]->setInputCorrespondences(tmp_corrs);
                    corr_rejectors[i]->getCorrespondences(correspondences_);

                    duna::logger::log_debug("rejected: %d / %d", correspondences_.size(), tmp_corrs->size());
                    // Modify input for the next iteration
                    if (i < corr_rejectors.size() - 1)
                        *tmp_corrs = correspondences_;
                }
            }

            if (correspondences_.size() < 4)
                duna::logger::log_debug("Too few correspondences! (%d / %d) ", correspondences_.size(), source_->size());
            overlap = (float)correspondences_.size() / (float)source_->size();
        }

        bool f(const Scalar *x, Scalar *f_x, unsigned int index) override
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

        inline void setMaximumCorrespondenceDistance(double distance)
        {
            maximum_corr_dist_ = distance;
        }

        inline float getOverlap() const
        {
            return overlap;
        }

        inline void addCorrespondenceRejector(pcl::registration::CorrespondenceRejector::Ptr rejector)
        {
            corr_rejectors.push_back(rejector);
        }

        inline void clearCorrespondenceRejectors()
        {
            corr_rejectors.clear();
        }

    protected:
        PointCloudSourceConstPtr source_;
        PointCloudTargetConstPtr target_;
        KdTreePtr kdtree_target_;
        PointCloudSourcePtr transformed_source_;
        pcl::Correspondences correspondences_;
        Eigen::Matrix<Scalar, 4, 4> transform_;
        pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar> corr_estimator_;
        std::vector<pcl::registration::CorrespondenceRejector::Ptr> corr_rejectors;

        float overlap;

        // Parameters
        double maximum_corr_dist_;
    };
}