#pragma once

#include <duna/registration/models/base_model.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar>
    struct Point2Plane3DOF
    {

        Point2Plane3DOF(const pcl::PointCloud<PointSource> &source_, const pcl::PointCloud<PointSource> &target_, const std::unordered_map<int, pcl::Normal> &target_normal_map_, const pcl::Correspondences &corrs_) : source(source_),
                                                                                                                                                                                                                        target(target_),
                                                                                                                                                                                                                        target_normal_map(target_normal_map_),
                                                                                                                                                                                                                        corrs(corrs_)
        {
        }
        void setup(const Scalar *x)
        {
            so3::convert3DOFParameterToMatrix(x, transform);
        }

        void operator()(const Scalar *x, Scalar *f_x, unsigned int index)
        {
            const PointSource &src_pt = source.points[corrs[index].index_query];
            const PointSource &tgt_pt = target.points[corrs[index].index_match];
            const pcl::Normal &tgt_normal = target_normal_map.at(corrs[index].index_query);

            Eigen::Matrix<Scalar, 4, 1> src_(src_pt.x, src_pt.y, src_pt.z, 1);
            Eigen::Matrix<Scalar, 4, 1> tgt_(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
            Eigen::Matrix<Scalar, 4, 1> tgt_normal_(tgt_normal.normal_x, tgt_normal.normal_y, tgt_normal.normal_z, 0);

            Eigen::Matrix<Scalar, 4, 1> warped_src_ = transform * src_;
            f_x[0] = (warped_src_ - tgt_).dot(tgt_normal_);
        }

        // void df(const Scalar *x, Scalar *jacobian, unsigned int index)
        // {
        // }

    private:
        const pcl::PointCloud<PointSource> &source;
        const pcl::PointCloud<PointTarget> &target;
        const std::unordered_map<int, pcl::Normal> &target_normal_map;
        const pcl::Correspondences &corrs;
        Eigen::Matrix<Scalar, 4, 4> transform;
    };
}