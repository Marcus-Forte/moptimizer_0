#include <duna/registration/transformation_estimationMAP.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar>
    void TransformationEstimatorMAP<PointSource, PointTarget, Scalar>::estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                                                                                   const pcl::PointCloud<PointTarget> &cloud_tgt,
                                                                                                   const pcl::Correspondences &correspondences,
                                                                                                   Matrix4 &transformation_matrix) const
    {
        auto optimizer = new duna::LevenbergMarquadt<Scalar, 6>;
        optimizer->setMaximumIterations(max_optimizator_iterations);

        auto cost = new duna::CostFunctionAnalytical<duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>, Scalar, 6, 1>(
            new duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>(cloud_src, cloud_tgt, correspondences), true);

        auto state_cost = new duna::CostFunctionRotationError<Scalar>;
        // std::cout << "Duna transform estimator\n";
        optimizer->addCost(cost);
        optimizer->addCost(state_cost);
        cost->setNumResiduals(correspondences.size());

        Eigen::Matrix<Scalar, 6, 1> x0;
        x0.setZero();

        optimizer->minimize(x0.data());

        so3::convert3DOFParameterToMatrix(x0.data(), transformation_matrix);

        if (overlap_)
            *overlap_ = (float)correspondences.size() / (float)cloud_src.size();

        delete optimizer;
        delete cost;
    }

    // template class TransformationEstimator3DOF<pcl::PointXYZ, pcl::PointXYZ, double>;
    // template class TransformationEstimator3DOF<pcl::PointXYZ, pcl::PointXYZ, float>;

    template class TransformationEstimatorMAP<pcl::PointNormal, pcl::PointNormal, double>;
    template class TransformationEstimatorMAP<pcl::PointNormal, pcl::PointNormal, float>;

    // template class TransformationEstimator3DOF<pcl::PointXYZI, pcl::PointXYZI, double>;
    // template class TransformationEstimator3DOF<pcl::PointXYZI, pcl::PointXYZI, float>;

    template class TransformationEstimatorMAP<pcl::PointXYZINormal, pcl::PointXYZINormal, double>;
    template class TransformationEstimatorMAP<pcl::PointXYZINormal, pcl::PointXYZINormal, float>;
}
