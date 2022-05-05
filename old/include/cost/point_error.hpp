#pragma once

#include "distance_functor.h"

/* This file contains implementations of distance metrics (i.e point to point distance, point to plane distance) 
   Client code could inherit distance_functor and implement their own distance.
*/

namespace duna 
{
    //Point to point distance.
    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class Point2Point : public DistanceFunctor<PointSource, PointTarget, Scalar>
    {          
        inline double operator()(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const PointTarget &tgt_pt) override
        {
            Eigen::Matrix<Scalar, 4, 1> tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
            Eigen::Matrix<Scalar, 4, 1> src(warped_src_pt[0], warped_src_pt[1], warped_src_pt[2], 0);
            return (src - tgt).norm(); // TODO norm vs norm² ?
        }
    };

    //Point to Plane distance. Require Normals in PointTarget type
    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class Point2Plane : public DistanceFunctor<PointSource, PointTarget, Scalar>
    {   
        inline double operator()(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const PointTarget &tgt_pt) override
        {
            Eigen::Matrix<Scalar, 4, 1> tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
            Eigen::Matrix<Scalar, 4, 1> src(warped_src_pt[0], warped_src_pt[1], warped_src_pt[2], 0);
            Eigen::Matrix<Scalar, 4, 1> tgt_normal(tgt_pt.normal_x, tgt_pt.normal_y, tgt_pt.normal_z, 0);
            return (src - tgt).dot(tgt_normal); // TODO norm vs norm² ?
        }
    };
}