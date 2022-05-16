#include <gtest/gtest.h>
#include <duna/stopwatch.hpp>
#include <duna/registration/registration.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_dual_quaternion.h>

using PointT = pcl::PointXYZ;
using PointCloutT = pcl::PointCloud<PointT>;

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
            throw std::runtime_error("Unable to laod test data 'bunny.pcd'");
        }

        std::cout << "Loaded : " << target->size() << " points\n";

        target_kdtree->setInputCloud(target);

        registration.setMaximumICPIterations(50);
        registration.setInputSource(source);
        registration.setInputTarget(target);
        registration.setTargetSearchMethod(target_kdtree);
        registration.setMaximumCorrespondenceDistance(10);
        registration.setMaximumOptimizerIterations(3);

        // pcl NL //

        pcl_icp.setInputSource(this->source);
        pcl_icp.setInputTarget(this->target);
        pcl_icp.setMaxCorrespondenceDistance(10);
        pcl_icp.setMaximumIterations(50);
        pcl_icp.setSearchMethodTarget(this->target_kdtree);

#ifndef NDEBUG
        pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
#endif
    }

    ~RegistrationPoint2Point()
    {
    }

protected:
    pcl::IterativeClosestPoint<PointT, PointT, Scalar> pcl_icp;
    typename pcl::registration::TransformationEstimation<PointT, PointT, Scalar>::Ptr estimation;
    duna::Registration<PointT, PointT, Scalar> registration;
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RegistrationPoint2Point, ScalarTypes);

TYPED_TEST(RegistrationPoint2Point, Translation)
{
    this->reference_transform(0, 3) = 1;
    this->reference_transform(1, 3) = 2;
    this->reference_transform(2, 3) = 3;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    utilities::Stopwatch timer;
    timer.tick();
    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
    std::string message("Duna ICP: " + std::to_string(this->registration.getFinalIterationsNumber()) + "/" + std::to_string(this->registration.getMaximumICPIterations()));
    timer.tock(message);

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    // this->estimation.reset(new pcl::registration::TransformationEstimationLM<PointT, PointT, TypeParam>);
    // this->estimation.reset(new pcl::registration::TransformationEstimationDualQuaternion<PointT,PointT,TypeParam>);
    this->estimation.reset(new pcl::registration::TransformationEstimationSVD<PointT, PointT, TypeParam>);
    this->pcl_icp.setTransformationEstimation(this->estimation);
    timer.tick();
    PointCloutT output;
    this->pcl_icp.align(output);
    timer.tock("PCL ICP");

    std::cerr << "PCL ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
}

TYPED_TEST(RegistrationPoint2Point, Rotation)
{

    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(0.2, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(0.8, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(0.6, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;
    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);
    utilities::Stopwatch timer;
    timer.tick();
    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
    std::string message("Duna ICP: " + std::to_string(this->registration.getFinalIterationsNumber()) + "/" + std::to_string(this->registration.getMaximumICPIterations()));
    timer.tock(message);

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    this->estimation.reset(new pcl::registration::TransformationEstimationLM<PointT, PointT, TypeParam>);
    this->pcl_icp.setTransformationEstimation(this->estimation);
    timer.tick();
    PointCloutT output;
    this->pcl_icp.align(output);
    timer.tock("PCL ICP");

    std::cerr << "PCL ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
}

TYPED_TEST(RegistrationPoint2Point, RotationPlusTranslation)
{

    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(0.8, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(0.5, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(0.2, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;
    this->reference_transform(0, 3) = -0.5;
    this->reference_transform(1, 3) = -0.2;
    this->reference_transform(2, 3) = 0.1;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);
    utilities::Stopwatch timer;
    timer.tick();
    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
    std::string message("Duna ICP: " + std::to_string(this->registration.getFinalIterationsNumber()) + "/" + std::to_string(this->registration.getMaximumICPIterations()));
    timer.tock(message);

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    this->estimation.reset(new pcl::registration::TransformationEstimationLM<PointT, PointT, TypeParam>);
    this->pcl_icp.setTransformationEstimation(this->estimation);
    timer.tick();
    PointCloutT output;
    this->pcl_icp.align(output);
    timer.tock("PCL ICP: ");

    std::cerr << "PCL ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
}

TYPED_TEST(RegistrationPoint2Point, Guess)
{
    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(0.0, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(0.0, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(0.0, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;
    this->reference_transform(0, 3) = 15;
    this->reference_transform(1, 3) = 15;
    this->reference_transform(2, 3) = 15;

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);
    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();
    
    Eigen::Matrix<TypeParam, 4, 4> guess = reference_transform_inverse;
    // Apply a small transform
    guess(0, 3) += 0.1;
    try
    {
        this->registration.align(guess);
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
}