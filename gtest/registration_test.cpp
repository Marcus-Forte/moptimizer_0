#include "registration.hpp"
#include "registration_cost.hpp"
#include "duna_log.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <gtest/gtest.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>
#ifndef TEST_DATA_DIR
#warning "NO 'TEST_DATA_DIR' DEFINED"
#define TEST_DATA_DIR "./"
#endif

#define MODEL_PARAM 6
#define MAXIT 25
#define MAXCORRDIST 2.0
// Optimization objects

using PointCloudT = pcl::PointCloud<pcl::PointXYZ>;
using VectorN = CostFunction<MODEL_PARAM>::VectorN;

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
  
    }

    virtual ~RegistrationTestClass(){}


protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
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

TEST_F(RegistrationTestClass, fixture)
{
    


}

TEST_F(RegistrationTestClass, Translation)
{

    
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.25, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

    // referece_transform.topLeftCorner<3,3>() = rot;
    // Translation
    referece_transform.col(3) = Eigen::Vector4f(1.1, 0.5, 0.5, 1);

    pcl::transformPointCloud(*target, *source, referece_transform);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(target);

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM> *cost = new RegistrationCost<MODEL_PARAM>(target->size(), &data);
    Registration<MODEL_PARAM> *registration = new Registration<MODEL_PARAM>(cost);

    registration->setMaxIt(MAXIT);

    cost->setMaxCorrDist(MAXCORRDIST);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);

    Eigen::Matrix4f final_reg_duna;
    so3::param2Matrix(x0, final_reg_duna);

    std::cerr << "Reference:\n"
              << referece_transform << std::endl;
    // std::cerr << "PCL:\n"
    //           << final_reg_pcl << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 3; i++)
    {
        EXPECT_NEAR(final_reg_duna(i, 3), -referece_transform(i, 3), 0.01);
    }
}

TEST(RegistrationTest, Rotation)
{
    PointCloudT::Ptr source(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT);

    if (pcl::io::loadPCDFile(TEST_DATA_DIR, *target) != 0)
    {
        std::cerr << "Make sure you run the rest at the binaries folder.\n";
        FAIL();
    }

    std::cout << "Loaded " << target->points.size() << " points\n";

    Eigen::MatrixX4f referece_transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

    referece_transform.col(3) = Eigen::Vector4f(0, 0, 0, 1);

    referece_transform.topLeftCorner(3, 3) = rot;

    pcl::transformPointCloud(*target, *source, referece_transform);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(target);

    // Prepare dataset
    datatype_t data;
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM> *cost = new RegistrationCost<MODEL_PARAM>(target->size(), &data);
    Registration<MODEL_PARAM> *registration = new Registration<MODEL_PARAM>(cost);

    registration->setMaxIt(MAXIT);
    cost->setMaxCorrDist(MAXCORRDIST);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);

    std::cerr << "Reference:\n"
              << referece_transform << std::endl;
    std::cerr << "final x0: " << x0 << std::endl;

    for (int i = 0; i < 3; i++)
    {
        EXPECT_NEAR(x0[i], -referece_transform(i, 3), 0.01);
    }

    FAIL();
}

TEST(RegistrationTest, DISABLED_Multipleinstances)
{
}