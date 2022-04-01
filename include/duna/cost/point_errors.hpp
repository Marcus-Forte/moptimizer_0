#pragma once

#include <Eigen/Core>
#include <pcl/point_types.h>



// Base Class

template <typename PointSource, typename PointTarget, typename Scalar=float>
class DistanceFunctor
{
    public:
    using Ptr = std::shared_ptr<DistanceFunctor<PointSource,PointTarget,Scalar>>;
    using ConstPtr = std::shared_ptr<const DistanceFunctor<PointSource,PointTarget,Scalar>>;

    DistanceFunctor(){}
    virtual ~DistanceFunctor(){}

    inline virtual double operator()(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const PointTarget &tgt_pt) = 0;
};

// Derived errors

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