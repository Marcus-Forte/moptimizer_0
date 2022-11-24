#include <pcl/registration/transformation_estimation.h>
#include <duna/map/cost_function_analytical_covariance.h>
#include <duna/map/cost_function_rotation_state.h>
#include <duna/levenberg_marquadt.h>
#include <duna/scan_matching/models/point2plane3dof.h>

#include <duna/logger.h>
#include <duna/so3.h>

namespace duna
{
    /* Wrapper Class around duna optimizer for PCL registration classes
       Uses Maximum a Posteriori approach with the given states.
    */
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class TransformationEstimatorMAP : public pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = pcl::shared_ptr<TransformationEstimatorMAP<PointSource, PointTarget, Scalar>>;
        using ConstPtr = pcl::shared_ptr<const TransformationEstimatorMAP<PointSource, PointTarget, Scalar>>;
        using Matrix4 = typename pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>::Matrix4;
        using StateVector = Eigen::Matrix<Scalar, 6, 1>;

        TransformationEstimatorMAP() = delete;

        TransformationEstimatorMAP(bool point2plane = false) : point2plane_(point2plane),
                                                               max_optimizator_iterations(3),
                                                               has_run_(false)
        {
            state_covariance_.setIdentity();
            measurement_covariance_ = 1;
        }
        virtual ~TransformationEstimatorMAP() = default;

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

        inline void setOverlapRef(float *overlap)
        {
            overlap_ = overlap;
        }

        // Set error state covariance (Matrix P).
        inline void setStateCovariance(typename duna::CostFunctionRotationError<Scalar>::StateMatrix &covariance)
        {
            state_covariance_ = covariance;
        }

        // Set measurement error covariance. Will be applied to all points (as a scalar multiplication to the cost)
        inline void setMeasurementCovariance(Scalar covariance)
        {
            measurement_covariance_ = covariance;
        }

        // Gets computed P_k = (I - KH)P_k-1
        typename duna::CostFunctionRotationError<Scalar>::StateMatrix getUpdatedCovariance() const
        {
            if (has_run_)
                return updated_covariance_;
            else
                throw std::runtime_error("Transformation not run yet!");
        }

    public:
        int max_optimizator_iterations;

    private:
        bool point2plane_;
        float *overlap_ = nullptr;
        Scalar measurement_covariance_;
        typename duna::CostFunctionRotationError<Scalar>::StateMatrix state_covariance_;

        mutable bool has_run_;
        mutable typename duna::CostFunctionRotationError<Scalar>::StateMatrix updated_covariance_;
    };
}