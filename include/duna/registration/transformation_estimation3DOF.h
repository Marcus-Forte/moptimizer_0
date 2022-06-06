#include <pcl/registration/transformation_estimation.h>
#include <duna/cost_function_numerical.h>
#include <duna/levenberg_marquadt.h>

#include <duna/registration/models/point2plane3dof.h>
#include <duna/logging.h>
#include <duna/so3.h>

namespace duna
{
    /* Wrapper Class around duna optimizer for PCL registration classes */
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class TransformationEstimator3DOF : public pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = std::shared_ptr<TransformationEstimator3DOF<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const TransformationEstimator3DOF<PointSource, PointTarget, Scalar>>;
        using Matrix4 = typename pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>::Matrix4;

        TransformationEstimator3DOF(bool point2plane = false) : m_point2plane(point2plane) {}
        virtual ~TransformationEstimator3DOF() = default;

        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    Matrix4 &transformation_matrix) const override
        {
            throw std::runtime_error("Unimplemented!");
        }
        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const pcl::Indices &indices_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    Matrix4 &transformation_matrix) const override
        {
            throw std::runtime_error("Unimplemented!");
        }

        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const pcl::Indices &indices_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    const pcl::Indices &indices_tgt,
                                    Matrix4 &transformation_matrix) const override
        {
            throw std::runtime_error("Unimplemented!");
        }
        // This is what pcl::icp end up using
        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    const pcl::Correspondences &correspondences,
                                    Matrix4 &transformation_matrix) const override;

    private:
        bool m_point2plane;
    };
}