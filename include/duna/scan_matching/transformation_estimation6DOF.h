#include <pcl/registration/transformation_estimation.h>
#include <duna/cost_function_numerical.h>
#include <duna/levenberg_marquadt.h>

#include <duna/scan_matching/models/point2point.h>
#include <duna/scan_matching/models/point2plane.h>
#include <duna/logger.h>
#include <duna/so3.h>

namespace duna
{
    /* Wrapper Class around duna optimizer for PCL registration classes */
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class TransformationEstimator6DOF : public pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = pcl::shared_ptr<TransformationEstimator6DOF<PointSource, PointTarget, Scalar>>;
        using ConstPtr = pcl::shared_ptr<const TransformationEstimator6DOF<PointSource, PointTarget, Scalar>>;
        using Matrix4 = typename pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>::Matrix4;

        // Signals point2plane metric
        TransformationEstimator6DOF(bool point2plane = false) : m_point2plane(point2plane) 
        {
            max_optimizator_iterations = 3;
        }

        virtual ~TransformationEstimator6DOF() = default;

        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    Matrix4 &transformation_matrix) const override
        {
            throw std::runtime_error("Unimplemented!");
        }
        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const std::vector<int> &indices_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    Matrix4 &transformation_matrix) const override
        {
            throw std::runtime_error("Unimplemented!");
        }

        void
        estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                    const std::vector<int> &indices_src,
                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                    const std::vector<int> &indices_tgt,
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

    public:
        int max_optimizator_iterations;

    private:
        bool m_point2plane;
    };
}