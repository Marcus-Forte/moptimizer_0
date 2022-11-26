#pragma once

#include "duna/so3.h"
#include "duna/model.h"
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    struct Point2Plane : public BaseModelJacobian<Scalar>
    {

        Point2Plane(const pcl::PointCloud<PointSource> &source_,
                    const pcl::PointCloud<PointSource> &target_,
                    const pcl::Correspondences &corrs_) : source(source_),
                                                          target(target_),
                                                          corrs(corrs_)
        {
        }
        virtual void init(const Scalar *x)
        {
            so3::convert6DOFParameterToMatrix(x, transform);
        }
        virtual void setup(const Scalar *x)
        {
            so3::convert6DOFParameterToMatrix(x, transform);
        }

        bool operator()(const Scalar *x, Scalar *f_x, unsigned int index)
        {
            const PointSource &src_pt = source.points[corrs[index].index_query];
            const PointTarget &tgt_pt = target.points[corrs[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);
            Eigen::Matrix<Scalar, 4, 1> tgt_normal_(static_cast<Scalar>(tgt_pt.normal_x), static_cast<Scalar>(tgt_pt.normal_y), static_cast<Scalar>(tgt_pt.normal_z), 0.0);

            Eigen::Matrix<Scalar, 4, 1> &&warped_src_ = transform * src_;

            f_x[0] = (warped_src_ - tgt_).dot(tgt_normal_);
            return true;
        }

        virtual void df(const Scalar *x, Scalar *jacobian, unsigned int index)
        {
            const PointSource &src_pt = source.points[corrs[index].index_query];
            const PointTarget &tgt_pt = target.points[corrs[index].index_match];

            Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x), static_cast<Scalar>(src_pt.y), static_cast<Scalar>(src_pt.z), 1.0);
            // Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x), static_cast<Scalar>(tgt_pt.y), static_cast<Scalar>(tgt_pt.z), 0.0);
            Eigen::Matrix<Scalar, 3, 1> tgt_normal_(static_cast<Scalar>(tgt_pt.normal_x), static_cast<Scalar>(tgt_pt.normal_y), static_cast<Scalar>(tgt_pt.normal_z));

            jacobian[0] = tgt_normal_[0];
            jacobian[1] = tgt_normal_[1];
            jacobian[2] = tgt_normal_[2];
            // Not sure why we multiply by two. Numerical Diff comparison suggested that.
            jacobian[3] = 2 * (tgt_normal_[2] * src_[1] - tgt_normal_[1] * src_[2]);
            jacobian[4] = 2 * (tgt_normal_[0] * src_[2] - tgt_normal_[2] * src_[0]);
            jacobian[5] = 2 * (tgt_normal_[1] * src_[0] - tgt_normal_[0] * src_[1]);
        }

    protected:
        const pcl::PointCloud<PointSource> &source;
        const pcl::PointCloud<PointTarget> &target;
        const pcl::Correspondences &corrs;
        Eigen::Matrix<Scalar, 4, 4> transform;
    };
}