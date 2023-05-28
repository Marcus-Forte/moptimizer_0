#pragma once

#include <pcl/common/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection.h>

#include "duna/logger.h"
#include "duna/model.h"
#include "duna/so3.h"

/* Unified point to plane 6DOF registration model. */
namespace duna {
template <typename PointSource, typename PointTarget, typename Scalar>
class ScanMatchingBase : public BaseModelJacobian<Scalar> {
 public:
  using Ptr = std::shared_ptr<ScanMatchingBase>;
  using PointCloudSource = pcl::PointCloud<PointSource>;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = pcl::PointCloud<PointTarget>;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  using KdTree = pcl::search::KdTree<PointTarget>;
  using KdTreePtr = typename KdTree::Ptr;

  ScanMatchingBase(PointCloudSourceConstPtr source,
                   PointCloudTargetConstPtr target, KdTreePtr kdtree_target)
      : source_(source),
        target_(target),
        kdtree_target_(kdtree_target),
        maximum_corr_dist_(std::numeric_limits<double>::max()) {
    if (!source_ || source_->size() == 0)
      duna::logger::log_error("No points at source cloud!");

    if (!target_ || target_->size() == 0)
      duna::logger::log_error("No points at target cloud!");

    if (!kdtree_target_) duna::logger::log_error("No target Kdtree!");

    corr_estimator_.setInputTarget(target_);
    corr_estimator_.setSearchMethodTarget(kdtree_target_,
                                          true);  // Never recompute.
    transformed_source_.reset(new PointCloudSource);
  }

  virtual ~ScanMatchingBase() = default;

  virtual void update(const Scalar *x) override {
    setup(x);
    pcl::transformPointCloud(*source_, *transformed_source_, transform_);

    // duna::logger::log_debug("Updating correspondences... @",
    // maximum_corr_dist_);

    corr_estimator_.setInputSource(transformed_source_);
    corr_estimator_.determineCorrespondences(correspondences_,
                                             maximum_corr_dist_);

    // duna::logger::log_debug("found: %d / %d", correspondences_.size(),
    // source_->size());

    // copy
    if (corr_rejectors.size()) {
      pcl::CorrespondencesPtr tmp_corrs(
          new pcl::Correspondences(correspondences_));
      for (int i = 0; i < corr_rejectors.size(); ++i) {
        duna::logger::log_debug("Using rejector: %s",
                                corr_rejectors[i]->getClassName().c_str());
        corr_rejectors[i]->setInputCorrespondences(tmp_corrs);
        corr_rejectors[i]->getCorrespondences(correspondences_);

        duna::logger::log_debug("Remaining: %d / %d", correspondences_.size(),
                                tmp_corrs->size());
        // Modify input for the next iteration
        if (i < corr_rejectors.size() - 1) *tmp_corrs = correspondences_;
      }
    }

    if (correspondences_.size() < 4)
      duna::logger::log_debug("Too few correspondences! (%d / %d) ",
                              correspondences_.size(), source_->size());
    overlap_ = (float)correspondences_.size() / (float)source_->size();
  }

  inline void setMaximumCorrespondenceDistance(double distance) {
    maximum_corr_dist_ = distance;
  }

  inline float getOverlap() const { return overlap_; }

  inline void addCorrespondenceRejector(
      pcl::registration::CorrespondenceRejector::Ptr rejector) {
    corr_rejectors.push_back(rejector);
  }

  inline void clearCorrespondenceRejectors() { corr_rejectors.clear(); }

  virtual void setup(const Scalar *x) override = 0;
  virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override = 0;
  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian,
                    unsigned int index) override {
    throw duna::Exception(
        "Non implemented jacobian model function `f_df` being used.");
    return false;
  }

 protected:
  PointCloudSourceConstPtr source_;
  PointCloudTargetConstPtr target_;
  KdTreePtr kdtree_target_;
  PointCloudSourcePtr transformed_source_;
  pcl::Correspondences correspondences_;
  Eigen::Matrix<Scalar, 4, 4> transform_;
  pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>
      corr_estimator_;
  std::vector<pcl::registration::CorrespondenceRejector::Ptr> corr_rejectors;

  float overlap_;

  // Parameters
  double maximum_corr_dist_;

  // Check if normal is usable.
  inline bool isNormalUsable(const PointTarget &point_with_normal) const {
    if (std::isnan(point_with_normal.normal_x) ||
        std::isnan(point_with_normal.normal_y) ||
        std::isnan(point_with_normal.normal_z)) {
      return false;
    }
    return true;
  }
};
}  // namespace duna