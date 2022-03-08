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

using VectorN = CostFunction<MODEL_PARAM>::VectorN;
using Vector3N = Eigen::Matrix<float, 3, 1>;

class RegistrationTestClassPoint2Plane : public testing::Test
{
public:
    RegistrationTestClassPoint2Plane()
    {
        source.reset(new PointCloudT);
        target.reset(new PointCloudT);
        target_normals.reset(new PointCloudNT);
        if (pcl::io::loadPCDFile(TEST_DATA_DIR, *target) != 0)
        {
            std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }

        pcl::copyPointCloud(*target, *target_normals);

        pcl::NormalEstimation<PointXYZ, PointNormal> ne;
        ne.setInputCloud(target);
        ne.setKSearch(5);
        ne.compute(*target_normals);

        referece_transform = Eigen::Matrix4f::Identity();
        kdtree.reset(new pcl::search::KdTree<PointXYZ>);
        kdtree->setInputCloud(target);

        kdtree_normals.reset(new pcl::search::KdTree<PointNormal>);
        kdtree_normals->setInputCloud(target_normals);

        
    }

    virtual ~RegistrationTestClassPoint2Plane() {}

protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
    PointCloudNT::Ptr target_normals;
    pcl::search::KdTree<PointXYZ>::Ptr kdtree;
    pcl::search::KdTree<PointNormal>::Ptr kdtree_normals;

    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>::dataset_t data;
    Eigen::MatrixX4f referece_transform;
};

TEST_F(RegistrationTestClassPoint2Plane, Translation6DOF)
{
    // Translation
    referece_transform.col(3) = Eigen::Vector4f(-0.5, 0.2, 0., 1);

    pcl::transformPointCloud(*target, *source, referece_transform);

    data.source = source;
    data.target = target_normals;
    data.tgt_kdtree = kdtree_normals;


    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal> *cost = new RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>(&data);
    Registration<MODEL_PARAM, PointXYZ, PointNormal> *registration = new Registration<MODEL_PARAM, PointXYZ, PointNormal>(cost);
    // registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
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

TEST_F(RegistrationTestClassPoint2Plane, Rotation6DOF)
{
    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    referece_transform.topLeftCorner(3, 3) = rot;

    
    pcl::transformPointCloud(*target, *source, referece_transform);

    // Prepare dataset
    data.source = source;
    data.target = target_normals;
    data.tgt_kdtree = kdtree_normals;


    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal> *cost = new RegistrationCost<MODEL_PARAM, PointXYZ, PointNormal>(&data);
    Registration<MODEL_PARAM, PointXYZ, PointNormal> *registration = new Registration<MODEL_PARAM, PointXYZ, PointNormal>(cost);
   
    registration->setMaxOptimizationIterations(3);
    registration->setMaxIcpIterations(MAXIT);
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
