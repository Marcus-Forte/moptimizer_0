#pragma once

#include <duna/registration/models/point2plane.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar>
    struct Point2Plane3DOF : public Point2Plane<PointSource,PointTarget,Scalar>
    {
        Point2Plane3DOF(const pcl::PointCloud<PointSource> &source_, const pcl::PointCloud<PointSource> &target_, const std::unordered_map<int, pcl::Normal> &target_normal_map_, const pcl::Correspondences &corrs_) :
        Point2Plane<PointSource,PointTarget,Scalar>(source_, target_, target_normal_map_,corrs_)
        {}
        
        void setup(const Scalar *x) override
        {
            so3::convert3DOFParameterToMatrix(x, transform);
        }

        private:
        using Point2Plane<PointSource,PointTarget,Scalar>::transform;
    };
}