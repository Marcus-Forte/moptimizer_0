#include <duna/registration/scan_matching_3dof.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    void ScanMatchingBase<PointSource, PointTarget, Scalar>::updateCorrespondences(pcl::CorrespondencesPtr correspondences)
    {
        // Estimate correspondences
        corr_estimator_->determineCorrespondences(*correspondences, max_corr_distance_);

        // Reject some
    }

    template class ScanMatchingBase<pcl::PointNormal, pcl::PointNormal, double>;
    template class ScanMatchingBase<pcl::PointNormal, pcl::PointNormal, float>;

    template class ScanMatchingBase<pcl::PointXYZINormal, pcl::PointXYZINormal, double>;
    template class ScanMatchingBase<pcl::PointXYZINormal, pcl::PointXYZINormal, float>;
} // namespace