#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

#include <duna/models/scan_matching.h>
#include <duna/levenberg_marquadt.h>
#include <duna/cost_function_numerical.h>
#include <duna/stopwatch.hpp>
#include <duna/levenberg_marquadt_dynamic.h>
#include <duna/cost_function_numerical_dynamic.h>
#include <duna/cost_function_analytical_dynamic.h>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RegistrationPoint2Point, ScalarTypes);

#define TOLERANCE 1e-2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

template <typename Scalar>
class RegistrationPoint2Point : public ::testing::Test
{
public:
    RegistrationPoint2Point()
    {
        source.reset(new PointCloutT);
        target.reset(new PointCloutT);
        target_kdtree.reset(new pcl::search::KdTree<PointT>);
        reference_transform.setIdentity();

        if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0)
        {
            throw std::runtime_error("Unable to load test data 'bunny.pcd'");
        }

        std::cout << "Loaded : " << target->size() << " points\n";

        target_kdtree->setInputCloud(target);

        this->optimizer.setMaximumIterations(150);

        duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);
    }

protected:
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
    Eigen::Matrix<Scalar, 4, 4> result_transform;
    duna::LevenbergMarquadt<Scalar, 6> optimizer;
};

// PCL fails this one
TYPED_TEST(RegistrationPoint2Point, Translation)
{
    // Arrange
    this->reference_transform(0, 3) = 0.1;
    this->reference_transform(1, 3) = 0.2;
    this->reference_transform(2, 3) = 0.3;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    typename duna::ScanMatching6DOFPoint2Point<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching6DOFPoint2Point<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));

    auto cost = new duna::CostFunctionNumerical<TypeParam, 6, 3>(scan_matcher_model, this->source->size());

    this->optimizer.addCost(cost);

    TypeParam x0[6] = {0};
    // Act
    this->optimizer.minimize(x0);
    so3::convert6DOFParameterToMatrix(x0, this->result_transform);

    // Assert

    std::cout << "Final X " << Eigen::Map<Eigen::Matrix<TypeParam, 6, 1>>(x0) << std::endl;
    std::cout << "Final Transform: " << this->result_transform << std::endl;
    std::cout << "Reference Transform: " << reference_transform_inverse << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(this->result_transform(i), reference_transform_inverse(i), TOLERANCE);
    }

    delete cost;
}

TYPED_TEST(RegistrationPoint2Point, RotationPlusTranslation)
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

    typename duna::ScanMatching6DOFPoint2Point<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching6DOFPoint2Point<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));

    auto cost = new duna::CostFunctionNumerical<TypeParam, 6, 3>(scan_matcher_model, this->source->size());

    this->optimizer.addCost(cost);

    TypeParam x0[6] = {0};
    // Act
    this->optimizer.minimize(x0);
    so3::convert6DOFParameterToMatrix(x0, this->result_transform);

    // Assert
    std::cout << "Final x: \n"
              << Eigen::Map<Eigen::Matrix<TypeParam, 6, 1>>(x0) << std::endl;
    std::cout << "Final Transform: \n"
              << this->result_transform << std::endl;
    std::cout << "Reference Transform: \n"
              << reference_transform_inverse << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
        EXPECT_NEAR(this->result_transform(i), reference_transform_inverse(i), TOLERANCE);

    delete cost;
}

TYPED_TEST(RegistrationPoint2Point, RotationPlusTranslationDynamic)
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

    typename duna::ScanMatching6DOFPoint2Point<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching6DOFPoint2Point<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));

    auto cost = new duna::CostFunctionAnalyticalDynamic<TypeParam>(scan_matcher_model, 6, 3, this->source->size());
    auto dyn_opt = duna::LevenbergMarquadtDynamic<TypeParam>(6);

    dyn_opt.addCost(cost);
    dyn_opt.setMaximumIterations(150);
    TypeParam x0[6] = {0};
    // Act
    dyn_opt.minimize(x0);
    so3::convert6DOFParameterToMatrix(x0, this->result_transform);

    // Assert
    std::cout << "Final x: \n"
              << Eigen::Map<Eigen::Matrix<TypeParam, 6, 1>>(x0) << std::endl;
    std::cout << "Final Transform: \n"
              << this->result_transform << std::endl;
    std::cout << "Reference Transform: \n"
              << reference_transform_inverse << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
        EXPECT_NEAR(this->result_transform(i), reference_transform_inverse(i), TOLERANCE);

    delete cost;
}