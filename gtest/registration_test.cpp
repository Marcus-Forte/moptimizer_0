#include "duna/registration.h"
#include "duna/cost/registration_cost.hpp"
#include "duna/duna_log.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <gtest/gtest.h>
#include <pcl/features/normal_3d.h>

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

using PointXYZ = pcl::PointXYZ;
using PointNormal = pcl::PointNormal;

using PointCloudT = pcl::PointCloud<PointXYZ>;
using PointCloudNT = pcl::PointCloud<PointNormal>;

using VectorN = CostFunction<MODEL_PARAM>::VectorN;
using Vector3N = Eigen::Matrix<float, 3, 1>;

class RegistrationTestClass : public testing::Test
{
public:
    RegistrationTestClass()
    {
        source.reset(new PointCloudT);
        target.reset(new PointCloudNT);
        if (pcl::io::loadPCDFile(TEST_DATA_DIR, *target) != 0)
        {
            std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }

        pcl::NormalEstimation<PointNormal, PointNormal> ne;
        ne.setInputCloud(target);
        ne.setKSearch(5);
        ne.compute(*target);

        referece_transform = Eigen::Matrix4f::Identity();
        kdtree.reset(new pcl::search::KdTree<PointNormal>);
        kdtree->setInputCloud(target);
    }

    virtual ~RegistrationTestClass() {}

protected:
    PointCloudT::Ptr source;
    PointCloudNT::Ptr target;
    pcl::search::KdTree<PointNormal>::Ptr kdtree;
    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>::dataset_t data;
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

TEST_F(RegistrationTestClass, Translation6DOFSimple)
{
    target->clear();

    for (int i = 0; i < 10; ++i)
    {
        PointNormal pt;
        pt.x = 10 * static_cast<float>(std::rand() / static_cast<float>(RAND_MAX));
        pt.y = 10 * static_cast<float>(std::rand() / static_cast<float>(RAND_MAX));
        pt.z = 10 * static_cast<float>(std::rand() / static_cast<float>(RAND_MAX));
        target->push_back(pt);
    }

    pcl::NormalEstimation<PointNormal, PointNormal> ne;
    ne.setInputCloud(target);
    ne.setKSearch(5);
    ne.compute(*target);

    std::cerr << *target << "\n";

    kdtree->setInputCloud(target);

    referece_transform.col(3) = Eigen::Vector4f(1, 2, 3, 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal> *cost = new RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>(&data);
    Registration<MODEL_PARAM, PointXYZ, PointNormal> *registration = new Registration<MODEL_PARAM, PointXYZ, PointNormal>(cost);

    // registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(50);
    registration->setMaxCorrespondenceDistance(20);

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

TEST_F(RegistrationTestClass, Translation6DOF)
{

    // Translation
    referece_transform.col(3) = Eigen::Vector4f(-0.5, 0.2, 0., 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal> *cost = new RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>(&data);
    Registration<MODEL_PARAM, PointXYZ, PointNormal> *registration = new Registration<MODEL_PARAM, PointXYZ, PointNormal>(cost);

    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(50);
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
          Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal> *cost = new RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>(&data);
    Registration<MODEL_PARAM, PointXYZ, PointNormal> *registration = new Registration<MODEL_PARAM, PointXYZ, PointNormal>(cost);

    // registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(150);
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
          Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;
    referece_transform.col(3) = Eigen::Vector4f(-0.5, -0.2, 0.1, 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal> *cost = new RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>(&data);
    Registration<MODEL_PARAM, PointXYZ, PointNormal> *registration = new Registration<MODEL_PARAM, PointXYZ, PointNormal>(cost);

    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(50);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

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
          Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<3, PointXYZ, PointNormal> *cost = new RegistrationCost<3, PointXYZ, PointNormal>(&data);
    Registration<3, PointXYZ, PointNormal> *registration = new Registration<3, PointXYZ, PointNormal>(cost);

    // registration->setMaxOptimizationIterations(3);
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

TEST_F(RegistrationTestClass, PointToNormal)
{
}

TEST_F(RegistrationTestClass, SeriesofCalls3DOF)
{
}