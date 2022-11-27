#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>

#include <duna/map/transformation_estimationMAP.h>
#include <duna/stopwatch.hpp>
#include <duna/models/scan_matching3dof.h>
#include <duna/cost_function_analytical.h>
#include <duna/cost_function_numerical.h>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScanMatching, ScalarTypes);

#define TOLERANCE 1e-2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);

    return RUN_ALL_TESTS();
}

template <typename Scalar>
class ScanMatching : public ::testing::Test
{
public:
    ScanMatching()
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

        // Arrange
        Eigen::Matrix<Scalar, 3, 3> rot;
        rot = Eigen::AngleAxis<Scalar>(1.5, Eigen::Matrix<Scalar, 3, 1>::UnitX()) *
              Eigen::AngleAxis<Scalar>(1.5, Eigen::Matrix<Scalar, 3, 1>::UnitY()) *
              Eigen::AngleAxis<Scalar>(3.4, Eigen::Matrix<Scalar, 3, 1>::UnitZ());

        reference_transform.topLeftCorner(3, 3) = rot;

        reference_transform_inverse = reference_transform.inverse();

        pcl::transformPointCloud(*target, *source, reference_transform);

        optimizer.setMaximumIterations(20);
    }

protected:
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
    Eigen::Matrix<Scalar, 4, 4> reference_transform_inverse;
    duna::LevenbergMarquadt<Scalar, 3> optimizer;
};

TYPED_TEST(ScanMatching, TestSimpleRegistrationNumericalCost)
{

    typename duna::ScanMatching3DOF<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching3DOF<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));
    // scan_matcher_model->setMaximumCorrespondenceDistance(0.1);

    // duna::CostFunctionAnalytical<TypeParam, 3, 1> *cost;
    auto cost = new duna::CostFunctionNumericalDiff<TypeParam, 3, 1>(scan_matcher_model, this->source->size());

    this->optimizer.addCost(cost);

    TypeParam x0[3];
    x0[0] = 0;
    x0[1] = 0;
    x0[2] = 0;
    // Act
    this->optimizer.minimize(x0);

    // Assert
    Eigen::Matrix<TypeParam, 4, 4> final_transform;
    so3::convert3DOFParameterToMatrix(x0, final_transform);

    delete cost;

    std::cout << "Final X " << Eigen::Map<Eigen::Matrix<TypeParam, 3, 1>>(x0) << std::endl;
    std::cout << "Final Transform: " << final_transform << std::endl;
    std::cout << "Reference Transform: " << this->reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform(i), this->reference_transform_inverse(i), TOLERANCE);
    }
}

TYPED_TEST(ScanMatching, TestSimpleRegistrationAnalyticalCost)
{

    typename duna::ScanMatching3DOF<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching3DOF<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));
    // scan_matcher_model->setMaximumCorrespondenceDistance(0.1);
    
    // duna::CostFunctionAnalytical<TypeParam, 3, 1> *cost;
    auto cost = new duna::CostFunctionAnalytical<TypeParam, 3, 1>(scan_matcher_model, this->source->size());

    this->optimizer.addCost(cost);

    TypeParam x0[3];
    x0[0] = 0;
    x0[1] = 0;
    x0[2] = 0;
    // Act

    this->optimizer.minimize(x0);

    // Assert
    Eigen::Matrix<TypeParam, 4, 4> final_transform;
    so3::convert3DOFParameterToMatrix(x0, final_transform);

    delete cost;

    std::cout << "Final X " << Eigen::Map<Eigen::Matrix<TypeParam, 3, 1>>(x0) << std::endl;
    std::cout << "Final Transform: " << final_transform << std::endl;
    std::cout << "Reference Transform: " << this->reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform(i), this->reference_transform_inverse(i), TOLERANCE);
    }
}

TYPED_TEST(ScanMatching, TestSimpleRegistrationWithInitialCondition)
{
    typename duna::ScanMatching3DOF<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    scan_matcher_model.reset(new duna::ScanMatching3DOF<PointT, PointT, TypeParam>(this->source, this->target, this->target_kdtree));

    auto cost = new duna::CostFunctionNumericalDiff<TypeParam, 3, 1>(scan_matcher_model, this->source->size());

    this->optimizer.addCost(cost);

    TypeParam x0[3];
    x0[0] = 1.02742;
    x0[1] = -1.33427;
    x0[2] = 1.12462;

    // Act
    this->optimizer.minimize(x0);

    // Assert
    Eigen::Matrix<TypeParam, 4, 4> final_transform;
    so3::convert3DOFParameterToMatrix(x0, final_transform);

    delete cost;

    std::cout << "Final X:\n"
              << Eigen::Map<Eigen::Matrix<TypeParam, 3, 1>>(x0) << std::endl;
    std::cout << "Final Transform: " << final_transform << std::endl;
    std::cout << "Reference Transform: " << this->reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform(i), this->reference_transform_inverse(i), TOLERANCE);
    }
}
