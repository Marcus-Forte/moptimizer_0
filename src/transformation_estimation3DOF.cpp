#include <duna/registration/transformation_estimation3DOF.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar>
    void TransformationEstimator3DOF<PointSource, PointTarget, Scalar>::estimateRigidTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                                                                                    const pcl::PointCloud<PointTarget> &cloud_tgt,
                                                                                                    const pcl::Correspondences &correspondences,
                                                                                                    Matrix4 &transformation_matrix) const
    {
        CostFunctionBase<Scalar> *cost;
        auto optimizer = new duna::LevenbergMarquadt<Scalar, 3>;
        optimizer->setMaximumIterations(max_optimizator_iterations);

        if (m_point2plane)
        {
            cost = new duna::CostFunctionAnalytical<duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>, Scalar, 3, 1>(
                new duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>(cloud_src, cloud_tgt, correspondences), true);
        }
        else
        {
            // cost = new duna::CostFunctionNumericalDiff<duna::Point2Point<PointSource, PointTarget, Scalar>, Scalar, 3, 1>(
            //     new duna::Point2Point<PointSource, PointTarget, Scalar>(cloud_src, cloud_tgt, correspondences), true);
            throw std::runtime_error("Not implemented.");
        }

        // std::cout << "Duna transform estimator\n";
        optimizer->addCost(cost);
        cost->setNumResiduals(correspondences.size());

        Eigen::Matrix<Scalar, 3, 1> x0;
        x0.setZero();

        optimizer->minimize(x0.data());

        so3::convert3DOFParameterToMatrix(x0.data(), transformation_matrix);

        if(overlap_)
            *overlap_ = (float) correspondences.size() / (float) cloud_src.size();

        delete optimizer;
        delete cost;
    }

    // template class TransformationEstimator3DOF<pcl::PointXYZ, pcl::PointXYZ, double>;
    // template class TransformationEstimator3DOF<pcl::PointXYZ, pcl::PointXYZ, float>;

    template class TransformationEstimator3DOF<pcl::PointNormal, pcl::PointNormal, double>;
    template class TransformationEstimator3DOF<pcl::PointNormal, pcl::PointNormal, float>;

    // template class TransformationEstimator3DOF<pcl::PointXYZI, pcl::PointXYZI, double>;
    // template class TransformationEstimator3DOF<pcl::PointXYZI, pcl::PointXYZI, float>;

    template class TransformationEstimator3DOF<pcl::PointXYZINormal, pcl::PointXYZINormal, double>;
    template class TransformationEstimator3DOF<pcl::PointXYZINormal, pcl::PointXYZINormal, float>;
}
