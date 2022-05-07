#ifndef REGISTRATION_MODEL_H
#define REGISTRATION_MODEL_H

#include <duna/model.h>
#include <pcl/point_cloud.h>
#include <duna/so3.h>

namespace duna
{

    template <typename PointSource, typename PointTarget>
    struct RegistrationModel
    {
        using Scalar = double;
        RegistrationModel(const pcl::PointCloud<PointSource> &source_, const pcl::PointCloud<PointSource> &target_, const pcl::Correspondences &correspondences_) : source(source_), target(target_), correspondences(correspondences_) {}

        inline virtual void setup(const Scalar *x)
        {
            so3::convert6DOFParameterToMatrix(x, transform);
        }

        inline void operator()(const Scalar *x, Scalar *f_x, const unsigned int index)
        {
            const PointSource &src_pt = source.points[correspondences[index].index_query];
            const PointSource &tgt_pt = target.points[correspondences[index].index_match];

            const Eigen::Matrix<Scalar, 4, 1> src(src_pt.x, src_pt.y, src_pt.z, 1);
            const Eigen::Matrix<Scalar, 4, 1> tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);

            Eigen::Matrix<Scalar, 4, 1> warped_src = transform * src;
                    
            warped_src[3] = 0;
            f_x[0] = (warped_src - tgt).norm();
        }

        const pcl::PointCloud<PointSource> &source;
        const pcl::PointCloud<PointSource> &target;
        const pcl::Correspondences &correspondences;
        Eigen::Matrix<Scalar, 4, 4> transform;
    };

}
#endif