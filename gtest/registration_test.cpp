#include "registration.hpp"
#include "cost/registration_cost.hpp"
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
using PointCloudNT = pcl::PointCloud<pcl::PointNormal>;
using VectorN = CostFunction<MODEL_PARAM>::VectorN;

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

        referece_transform = Eigen::Matrix4f::Identity();
    }

    virtual ~RegistrationTestClass() {}

protected:
    PointCloudT::Ptr source;
    PointCloudNT::Ptr target;
    RegistrationCost<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal>::datatype_t data;
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

TEST_F(RegistrationTestClass, Translation)
{
   

    // Translation
    referece_transform.col(3) = Eigen::Vector4f(1.1, 0.5, 0.5, 1);


    pcl::copyPointCloud(*target,*source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree(new pcl::search::KdTree<pcl::PointNormal>);
    kdtree->setInputCloud(target);

    

    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal> *cost = new RegistrationCost<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal>(&data);
    Registration<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal> *registration = new Registration<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal>(cost);

    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(20);
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

TEST_F(RegistrationTestClass, Rotation)
{

    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

   referece_transform.col(3) = Eigen::Vector4f(0.1,0,0,1);

    referece_transform.topLeftCorner(3, 3) = rot;

    pcl::copyPointCloud(*target,*source);
    pcl::transformPointCloud(*source, *source, referece_transform);

    pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree(new pcl::search::KdTree<pcl::PointNormal>);
    kdtree->setInputCloud(target);

    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;

    RegistrationCost<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal> *cost = new RegistrationCost<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal>(&data);
    Registration<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal> *registration = new Registration<MODEL_PARAM,pcl::PointXYZ,pcl::PointNormal>(cost);

    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(100);
    registration->setMaxCorrespondenceDistance(2);

    VectorN x0;
    x0.setZero();

    try{
    registration->minimize(x0);
    } catch (std::runtime_error& er){
            
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

TEST(RegistrationTestClass, DISABLED_Rotation3DOF)
{
}


TEST(RegistrationTestClass, DISABLED_Multipleinstances)
{
}