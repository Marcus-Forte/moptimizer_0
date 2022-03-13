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

#define DOF6 6
#define DOF3 3
#define MAXIT 50
#define MAXCORRDIST 2.0
// Optimization objects

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i)
    {
        printf("arg: %2d = %s\n", i, argv[i]);
    }

    return RUN_ALL_TESTS();
}

using PointXYZ = pcl::PointXYZ;
using PointNormal = pcl::PointNormal;

using PointCloudT = pcl::PointCloud<PointXYZ>;
using PointCloudNT = pcl::PointCloud<PointNormal>;

using VectorN = CostFunction<DOF6>::VectorN;
using Vector3N = Eigen::Matrix<float, 3, 1>;

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

        reference_transform = Eigen::Matrix4f::Identity();
        kdtree.reset(new pcl::search::KdTree<PointXYZ>);
        kdtree->setInputCloud(target);
    }

    virtual ~RegistrationTestClass() {}

protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
    pcl::search::KdTree<PointXYZ>::Ptr kdtree;

    Eigen::MatrixX4f reference_transform;
};

TEST_F(RegistrationTestClass, Translation6DOFSimple)
{
    target->clear();

    for (int i = 0; i < 10; ++i)
    {
        PointXYZ pt;
        pt.x = 10 * static_cast<float>(std::rand() / static_cast<float>(RAND_MAX));
        pt.y = 10 * static_cast<float>(std::rand() / static_cast<float>(RAND_MAX));
        pt.z = 10 * static_cast<float>(std::rand() / static_cast<float>(RAND_MAX));
        target->push_back(pt);
    }

    std::cerr << *target << "\n";

    kdtree->setInputCloud(target);

    reference_transform.col(3) = Eigen::Vector4f(1, 2, 3, 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, reference_transform);

    RegistrationCost<DOF6, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF6, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF6, PointXYZ, PointXYZ>(&data);
    Registration<DOF6, PointXYZ, PointXYZ> *registration = new Registration<DOF6, PointXYZ, PointXYZ>(cost);

    // registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(20);


    registration->minimize();

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    // std::cerr << "PCL:\n"
    //           << final_reg_pcl << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, Translation6DOF)
{

    // Translation
    reference_transform.col(3) = Eigen::Vector4f(-0.5, 0.3, 0.2, 1);

    pcl::transformPointCloud(*target, *source, reference_transform);

    RegistrationCost<DOF6, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF6, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF6, PointXYZ, PointXYZ>(&data);
    Registration<DOF6, PointXYZ, PointXYZ> *registration = new Registration<DOF6, PointXYZ, PointXYZ>(cost);
    // registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);


    registration->minimize();

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << reference_transform << std::endl;
    // std::cerr << "PCL:\n"
    //           << final_reg_pcl << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, Rotation6DOF)
{

    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF6, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF6, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF6, PointXYZ, PointXYZ>(&data);
    Registration<DOF6, PointXYZ, PointXYZ> *registration = new Registration<DOF6, PointXYZ, PointXYZ>(cost);

    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(2);



    try
    {
        registration->minimize();
    }
    catch (std::runtime_error &er)
    {
    }
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, RotationPlusTranslation6DOF)
{

    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Eigen::Vector4f(-0.5, -0.2, 0.1, 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF6, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF6, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF6, PointXYZ, PointXYZ>(&data);
    Registration<DOF6, PointXYZ, PointXYZ> *registration = new Registration<DOF6, PointXYZ, PointXYZ>(cost);
    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(5);


    registration->minimize();

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, Rotation3DOF)
{
    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, reference_transform);

    RegistrationCost<DOF3, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF3, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF3, PointXYZ, PointXYZ>(&data);
    Registration<DOF3, PointXYZ, PointXYZ> *registration = new Registration<DOF3, PointXYZ, PointXYZ>(cost);
    // registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(2);


    try
    {
        registration->minimize();
    }
    catch (std::runtime_error &er)
    {
    }
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClass, Tough6DOF)
{

    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Eigen::Vector4f(-0.9, -0.5, 0.5, 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF6, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF6, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF6, PointXYZ, PointXYZ>(&data);
    Registration<DOF6, PointXYZ, PointXYZ> *registration = new Registration<DOF6, PointXYZ, PointXYZ>(cost);
    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(5);

    registration->minimize();

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }

}


TEST_F(RegistrationTestClass, Guess6DOF)
{

    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Eigen::Vector4f(-15.9, -35.5, 12.9, 1);

    pcl::copyPointCloud(*target, *source);
    pcl::transformPointCloud(*source, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF6, PointXYZ, PointXYZ>::dataset_t data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = kdtree;

    RegistrationCost<DOF6, PointXYZ, PointXYZ> *cost = new RegistrationCost<DOF6, PointXYZ, PointXYZ>(&data);
    Registration<DOF6, PointXYZ, PointXYZ> *registration = new Registration<DOF6, PointXYZ, PointXYZ>(cost);
    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(10);

    Eigen::Matrix4f guess = reference_transform.inverse();

    // Apply a small transform to move guess a bit further from reference
    guess(0,3) += 0.5;
    guess(1,3) += 0.2;
    guess(2,3) += 0.1;
    registration->minimize(guess);

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference (inverse):\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }

}