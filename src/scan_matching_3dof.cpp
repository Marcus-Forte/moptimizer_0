#include <duna/registration/scan_matching_3dof.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar>
    void ScanMatching3DOF<PointSource, PointTarget, Scalar>::match(const Matrix4 &guess)
    {
        if (!source_)
        {
            logger_.log(L_ERROR, "No source point cloud set.");
            return;
        }
        if (!target_)
        {
            logger_.log(L_ERROR, "No target input cloud set.");
            return;
        }

        if (!target_tree_)
        {
            logger_.log(L_ERROR, "No target kdtree set.");
            return;
        }

        // Setup;
        corr_estimator_->setSearchMethodTarget(target_tree_);
        corr_estimator_->setInputTarget(target_);

        // PointCloudSourcePtr input_transformed = pcl::make_shared<PointCloudSource>();
        PointCloudSourcePtr input_transformed(new PointCloudSource);

        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);\

        final_transform_ = guess;

        Eigen::Matrix<Scalar, 4, 4> iteration_transform;
        pcl::transformPointCloud(*source_, *input_transformed, final_transform_);

        int curr_it = 0;
        OptimizationStatus status;
        Scalar x0_[3];
        do
        {
            logger_.log(L_DEBUG, "Reg.IT %d/%d", curr_it, max_num_iterations_);
            corr_estimator_->setInputSource(input_transformed);

            logger_.log(L_DEBUG, "Updating Correspondences at %f max dist", max_corr_distance_);
            ScanMatchingBase<PointSource, PointTarget, Scalar>::updateCorrespondences(correspondences);

            overlap_ = (float)correspondences->size() / (float)source_->size();
            logger_.log(L_DEBUG, "Found: %d / %d", correspondences->size(), source_->size());

            CostFunctionBase<Scalar> *cost = new duna::CostFunctionAnalytical<duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>, Scalar, 3, 1>(
                new duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>(*input_transformed, *target_, *correspondences), true);

            cost->setNumResiduals(correspondences->size());

            optimizer_->setMaximumIterations(max_num_opt_iterations_);
            optimizer_->addCost(cost);
            x0_[0] = 0;
            x0_[1] = 0;
            x0_[2] = 0;
            status = optimizer_->minimize(x0_);

            so3::convert3DOFParameterToMatrix<Scalar>(x0_, iteration_transform);
            pcl::transformPointCloud(*input_transformed, *input_transformed, iteration_transform);

            final_transform_ = iteration_transform * final_transform_;

            // std::cout << "status = " << status << std::endl;
            // logger_.log(L_DEBUG, "x0_: (%f, %f, %f)", x0[0], x0[1], x0[2]);

            // x0[0] -= x0_[0];
            // x0[1] -= x0_[1];
            // x0[2] -= x0_[2];

            optimizer_->clearCosts();
            delete cost;

            curr_it++;

        } while (curr_it < max_num_iterations_ && status != OptimizationStatus::CONVERGED);

        logger_.log(L_DEBUG, "Converged");
    }
    template <typename PointSource, typename PointTarget, typename Scalar>
    void ScanMatching3DOF<PointSource, PointTarget, Scalar>::match(Scalar *x0)
    {
        if (!source_)
        {
            logger_.log(L_ERROR, "No source point cloud set.");
            return;
        }
        if (!target_)
        {
            logger_.log(L_ERROR, "No target input cloud set.");
            return;
        }

        if (!target_tree_)
        {
            logger_.log(L_ERROR, "No target kdtree set.");
            return;
        }

        // Setup;
        corr_estimator_->setSearchMethodTarget(target_tree_);
        corr_estimator_->setInputTarget(target_);

        // PointCloudSourcePtr input_transformed = pcl::make_shared<PointCloudSource>();
        PointCloudSourcePtr input_transformed(new PointCloudSource);

        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);

        Eigen::Matrix<Scalar, 4, 4> iteration_transform;
        so3::convert3DOFParameterToMatrix<Scalar>(x0, final_transform_);
        pcl::transformPointCloud(*source_, *input_transformed, final_transform_);

        int curr_it = 0;
        OptimizationStatus status;
        Scalar x0_[3];
        do
        {
            logger_.log(L_DEBUG, "Reg.IT %d/%d", curr_it, max_num_iterations_);
            corr_estimator_->setInputSource(input_transformed);

            logger_.log(L_DEBUG, "Updating Correspondences at %f max dist", max_corr_distance_);
            ScanMatchingBase<PointSource, PointTarget, Scalar>::updateCorrespondences(correspondences);

            overlap_ = (float)correspondences->size() / (float)source_->size();
            logger_.log(L_DEBUG, "Found: %d / %d", correspondences->size(), source_->size());

            CostFunctionBase<Scalar> *cost = new duna::CostFunctionAnalytical<duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>, Scalar, 3, 1>(
                new duna::Point2Plane3DOF<PointSource, PointTarget, Scalar>(*input_transformed, *target_, *correspondences), true);

            cost->setNumResiduals(correspondences->size());

            optimizer_->setMaximumIterations(max_num_opt_iterations_);
            optimizer_->addCost(cost);
            x0_[0] = 0;
            x0_[1] = 0;
            x0_[2] = 0;
            status = optimizer_->minimize(x0_);

            so3::convert3DOFParameterToMatrix<Scalar>(x0_, iteration_transform);
            pcl::transformPointCloud(*input_transformed, *input_transformed, iteration_transform);

            final_transform_ = iteration_transform * final_transform_;

            // std::cout << "status = " << status << std::endl;
            // logger_.log(L_DEBUG, "x0_: (%f, %f, %f)", x0[0], x0[1], x0[2]);

            // x0[0] -= x0_[0];
            // x0[1] -= x0_[1];
            // x0[2] -= x0_[2];

            optimizer_->clearCosts();
            delete cost;

            curr_it++;

        } while (curr_it < max_num_iterations_ && status != OptimizationStatus::CONVERGED);

        logger_.log(L_DEBUG, "Converged");
    }

    template class ScanMatching3DOF<pcl::PointNormal, pcl::PointNormal, double>;
    template class ScanMatching3DOF<pcl::PointNormal, pcl::PointNormal, float>;

    template class ScanMatching3DOF<pcl::PointXYZINormal, pcl::PointXYZINormal, double>;
    template class ScanMatching3DOF<pcl::PointXYZINormal, pcl::PointXYZINormal, float>;
} // namespace