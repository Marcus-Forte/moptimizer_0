#pragma once

#include <duna/models/scan_matching_point2plane_6dof.h>

/* Unified point to plane 3DOF registration model. */
namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    class ScanMatching3DOFPoint2Plane : public ScanMatching6DOFPoint2Plane<PointSource, PointTarget, Scalar>
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
                         KdTreePtr kdtree_target) : ScanMatching6DOFPoint2Plane<PointSource, PointTarget, Scalar>(source, target, kdtree_target)
        {}

        virtual ~ScanMatching3DOFPoint2Plane() = default;

        void setup(const Scalar *x) override
        {
            so3::convert3DOFParameterToMatrix(x, transform_);
        }

        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) override 
        {
            jacobian[0] = 1;
            return true;
        }

    protected:
        using ScanMatchingBase<PointSource, PointTarget,Scalar>::transform_;
    };
}