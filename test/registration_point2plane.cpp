#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

#include <duna/scan_matching/transformation_estimation6DOF.h>
#include <duna/models/scan_matching3dof.h>
#include <duna/models/scan_matching6dof.h>
#include <duna/stopwatch.hpp>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RegistrationPoint2Plane, ScalarTypes);

#define TOLERANCE 1e-2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

/* This class tests the use of Duna optimizer as a transform estimator */

template <typename Scalar>
class RegistrationPoint2Plane : public ::testing::Test
{
public:
    RegistrationPoint2Plane()
    {
        source.reset(new PointCloutT);
        target.reset(new PointCloutT);
        target_kdtree.reset(new pcl::search::KdTree<PointT>);
        reference_transform.setIdentity();

        if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0)
        {
            throw std::runtime_error("Unable to laod test data 'bunny.pcd'");
        }

        std::cout << "Loaded : " << target->size() << " points\n";

        target_kdtree->setInputCloud(target);

        pcl::NormalEstimation<PointT, PointT> ne;
        ne.setInputCloud(target);
        ne.setSearchMethod(target_kdtree);
        ne.setKSearch(10);
        ne.compute(*target);

        pcl_icp.setInputTarget(target);
        pcl_icp.setMaxCorrespondenceDistance(10);
        pcl_icp.setMaximumIterations(100);
        pcl_icp.setSearchMethodTarget(target_kdtree);
        pcl_icp.setTransformationEpsilon(1e-4);
        
        pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    }

protected:
    pcl::IterativeClosestPoint<PointT, PointT, Scalar> pcl_icp;
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
};

// PCL fails this one
TYPED_TEST(RegistrationPoint2Plane, Translation)
{
    this->reference_transform(0, 3) = 0.5;
    this->reference_transform(1, 3) = 0.2;
    this->reference_transform(2, 3) = 0.3;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    // Instantiate estimators
    typename pcl::registration::TransformationEstimationPointToPlane<PointT, PointT, TypeParam>::Ptr pcl_transform(new pcl::registration::TransformationEstimationPointToPlane<PointT, PointT, TypeParam>);
    typename duna::TransformationEstimator6DOF<PointT, PointT, TypeParam>::Ptr duna_transform(new duna::TransformationEstimator6DOF<PointT, PointT, TypeParam>(true));
    PointCloutT output;

    this->pcl_icp.setInputSource(this->source);
    this->pcl_icp.setTransformationEstimation(pcl_transform);
    utilities::Stopwatch timer;
    timer.tick();
    this->pcl_icp.align(output);
    Eigen::Matrix<TypeParam, 4, 4> final_transform_pcl = this->pcl_icp.getFinalTransformation();
    timer.tock("PCL LM");

    std::cerr << "PCL ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;
    duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);
    duna::LevenbergMarquadt<TypeParam, 6> optimizer;
    typename duna::ScanMatching6DOF<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching6DOF<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));
    scan_matcher_model->setMaximumCorrespondenceDistance(10);

    auto cost = new duna::CostFunctionNumericalDiff<TypeParam, 6, 1>(scan_matcher_model, this->source->size());

    optimizer.addCost(cost);

    TypeParam x0[6] = {0,0,0,0,0,0};
    // Act
    optimizer.minimize(x0);

    Eigen::Matrix<TypeParam, 4, 4> final_transform_duna;
    so3::convert6DOFParameterToMatrix(x0, final_transform_duna);


    std::cerr
        << "PCL/DUNA ICP: \n";
    std::cerr << final_transform_duna << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform_duna(i), reference_transform_inverse(i), TOLERANCE);
    }
}

TYPED_TEST(RegistrationPoint2Plane, RotationPlusTranslation)
{
    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(0.3, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(0.4, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(0.5, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;
    this->reference_transform(0, 3) = 0.5;
    this->reference_transform(1, 3) = 0.2;
    this->reference_transform(2, 3) = 0.3;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    // Instantiate estimators
    typename pcl::registration::TransformationEstimationPointToPlane<PointT, PointT, TypeParam>::Ptr pcl_transform(new pcl::registration::TransformationEstimationPointToPlane<PointT, PointT, TypeParam>);
    typename duna::TransformationEstimator6DOF<PointT, PointT, TypeParam>::Ptr duna_transform(new duna::TransformationEstimator6DOF<PointT, PointT, TypeParam>(true));
    PointCloutT output;

    this->pcl_icp.setInputSource(this->source);
    this->pcl_icp.setTransformationEstimation(pcl_transform);
    utilities::Stopwatch timer;
    timer.tick();
    this->pcl_icp.align(output);
    Eigen::Matrix<TypeParam, 4, 4> final_transform_pcl = this->pcl_icp.getFinalTransformation();
    timer.tock("PCL LM");

    std::cerr << "PCL ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    this->pcl_icp.setTransformationEstimation(duna_transform);
    timer.tick();
    this->pcl_icp.align(output);
    Eigen::Matrix<TypeParam, 4, 4> final_transform_duna = this->pcl_icp.getFinalTransformation();
    timer.tock("DUNA LM");

    std::cerr
        << "PCL/DUNA ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform_duna(i), reference_transform_inverse(i), TOLERANCE);
    }
}