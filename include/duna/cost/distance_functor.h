#pragma once

#include <Eigen/Core>
#include <pcl/point_types.h>

/*  Base class for implementing distance functions for cost
    
*/

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar=float>
    class DistanceFunctor
    {
        public:
        using Ptr = std::shared_ptr<DistanceFunctor<PointSource,PointTarget,Scalar>>;
        using ConstPtr = std::shared_ptr<const DistanceFunctor<PointSource,PointTarget,Scalar>>;

        DistanceFunctor() = default;
        virtual ~DistanceFunctor() = default;

        inline virtual double operator()(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const PointTarget &tgt_pt) = 0;
    };
}