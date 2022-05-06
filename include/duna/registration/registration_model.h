#ifndef REGISTRATION_MODEL_H
#define REGISTRATION_MODEL_H

#include <duna/model.h>
#include <pcl/point_cloud.h>
#include <duna/so3.h>

namespace duna
{

    template <typename PointSource, typename PointTarget>
    struct RegistrationModel : public Model<float>
    {
        RegistrationModel(const pcl::PointCloud<PointSource> &source_, const pcl::PointCloud<PointSource> &target_, const pcl::Correspondences &correspondences_) : source(source_), target(target_), correspondences(correspondences_)
        {
        }
        inline virtual void setup(const float *x) override
        {
            so3::convert6DOFParameterToMatrix(x, transform);
        }

        inline void operator()(const float *x, float *f_x, const unsigned int index) override
        {
            const PointSource &src_pt = source.points[correspondences[index].index_query];
            const PointSource &tgt_pt = target.points[correspondences[index].index_match];

            const Eigen::Vector4f warped_src_pt = transform * src_pt.getVector4fMap();


            const Eigen::Vector4f tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
            const Eigen::Vector4f src(warped_src_pt[0],warped_src_pt[1],warped_src_pt[2],0);
            
            f_x[0] = (src - tgt).norm();
        }

        const pcl::PointCloud<PointSource> &source;
        const pcl::PointCloud<PointSource> &target;
        const pcl::Correspondences &correspondences;
        Eigen::Matrix<float, 4, 4> transform;
    };

}
#endif