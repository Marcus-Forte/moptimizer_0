#include "registration.h"
#include "cost/registration_cost.hpp"
#include "duna_log.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <gtest/gtest.h>

// #include <pcl/registration/icp.h>
// #include <pcl/registration/transformation_estimation_lm.h>
#ifndef TEST_DATA_DIR
#warning "NO 'TEST_DATA_DIR' DEFINED"
#define TEST_DATA_DIR "./"
#endif

#define MODEL_PARAM 6
#define MAXIT 10
#define MAXCORRDIST 2.0
// Optimization objects

using PointCloudT = pcl::PointCloud<pcl::PointXYZ>;
using VectorN = CostFunction<MODEL_PARAM>::VectorN;
using Vector3N = Eigen::Matrix<float,3,1>;

class RegistrationTestClass : public testing::Test
{
public:
    RegistrationTestClass()
    {
        source.reset(new PointCloudT);
        target.reset(new PointCloudT);
        if (pcl::io::loadPCDFile(TEST_DATA_DIR, *target) != 0)
        {
            std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }

        referece_transform = Eigen::Matrix4f::Identity();
        kdtree.reset(new pcl::search::KdTree<pcl::PointXYZ>);
        kdtree->setInputCloud(target);
    }

    virtual ~RegistrationTestClass() {}

protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
    datatype_t data;
    Eigen::MatrixX4f referece_transform;
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i)
    {
        printf("arg: %2d = %s\n", i, argv[i]);
    }

    return RUN_ALL_TESTS();
}

TEST_F(RegistrationTestClass, Translation6DOF)
{

    // Translation
    referece_transform.col(3) = Eigen::Vector4f(1.0, 0., 0., 1);
    pcl::transformPointCloud(*target, *source, referece_transform);

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM> *cost = new RegistrationCost<MODEL_PARAM>(&data);
    Registration<MODEL_PARAM> *registration = new Registration<MODEL_PARAM>(cost);

    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(100);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << referece_transform << std::endl;
    // std::cerr << "PCL:\n"
    //           << final_reg_pcl << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(referece_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, Rotation6DOF)
{

    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;

    pcl::transformPointCloud(*target, *source, referece_transform);

    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM> *cost = new RegistrationCost<MODEL_PARAM>(&data);
    Registration<MODEL_PARAM> *registration = new Registration<MODEL_PARAM>(cost);

    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(100);
    registration->setMaxCorrespondenceDistance(2);

    VectorN x0;
    x0.setZero();

    try
    {
        registration->minimize(x0);
    }
    catch (std::runtime_error &er)
    {
    }
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << referece_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(referece_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, RotationPlusTranslation6DOF)
{

    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;
    referece_transform.col(3) = Eigen::Vector4f(0.1, -0.2, 0., 1);

    pcl::transformPointCloud(*target, *source, referece_transform);

    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM> *cost = new RegistrationCost<MODEL_PARAM>(&data);
    Registration<MODEL_PARAM> *registration = new Registration<MODEL_PARAM>(cost);

    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(100);
    registration->setMaxCorrespondenceDistance(2);

    VectorN x0;
    x0.setZero();

    try
    {
        registration->minimize(x0);
    }
    catch (std::runtime_error &er)
    {
    }
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << referece_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(referece_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, Rotation3DOF)
{
    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;

    pcl::transformPointCloud(*target, *source, referece_transform);

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<3> *cost = new RegistrationCost<3>(&data);
    Registration<3> *registration = new Registration<3>(cost);

    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(100);
    registration->setMaxCorrespondenceDistance(2);

    Vector3N x0;
    x0.setZero();

    try
    {
        registration->minimize(x0);
    }
    catch (std::runtime_error &er)
    {
    }
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << referece_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(referece_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST(RegistrationTestClass, DISABLED_PointNormals)
{
}

TEST(RegistrationTestClass, DISABLED_Multipleinstances)
{
}