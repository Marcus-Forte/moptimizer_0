#include "duna/registration.h"
#include "duna/cost/registration_cost.hpp"
#include "duna/duna_log.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <gtest/gtest.h>
#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
using namespace duna;
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

class RegistrationTestClassPoint2Plane : public testing::Test
{
public:
    RegistrationTestClassPoint2Plane()
    {
        source.reset(new PointCloudT);
        target.reset(new PointCloudT);
        target_normals.reset(new PointCloudNT);
        if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0)
        {
            std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }

        pcl::copyPointCloud(*target, *target_normals);

        pcl::NormalEstimation<PointXYZ, PointNormal> ne;
        ne.setInputCloud(target);
        ne.setKSearch(5);
        ne.compute(*target_normals);

        reference_transform = Eigen::Matrix4f::Identity();
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

    Eigen::Matrix4f reference_transform;
};

TEST_F(RegistrationTestClassPoint2Plane, Translation6DOF)
{
    // Translation
    reference_transform.col(3) = Eigen::Vector4f(-0.5, 0.3, 0.2, 1); // Much faster than point 2 point

    pcl::transformPointCloud(*target, *source, reference_transform);

    RegistrationCost<DOF6, PointXYZ, PointNormal>::dataset_t data;
    data.source = source;
    data.target = target_normals;
    data.tgt_search_method = kdtree_normals;

    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<DOF6, PointXYZ, PointNormal> *cost = new RegistrationCost<DOF6, PointXYZ, PointNormal>(&data);
    Registration<DOF6, PointXYZ, PointNormal> *registration = new Registration<DOF6, PointXYZ, PointNormal>(cost);
    registration->setMaxOptimizationIterations(2);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    // PCL ICP
    pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
    pcl::IterativeClosestPoint<PointNormal, PointNormal> icp; // PCL icp object is awkwardly templated
    PointCloudNT::Ptr source_normals(new PointCloudNT);
    pcl::copyPointCloud(*source, *source_normals);
    icp.setInputSource(source_normals);
    icp.setInputTarget(target_normals);
    icp.setMaxCorrespondenceDistance(MAXCORRDIST);
    icp.setMaximumIterations(MAXIT);
    icp.setSearchMethodTarget(kdtree_normals);

    pcl::registration::TransformationEstimationPointToPlane<PointNormal, PointNormal>::Ptr te(new pcl::registration::TransformationEstimationPointToPlane<PointNormal, PointNormal>);
    // pcl::registration::TransformationEstimationPointToPlaneLLS<PointNormal,PointNormal>::Ptr te
    // (new pcl::registration::TransformationEstimationPointToPlaneLLS<PointNormal,PointNormal>);
    icp.setTransformationEstimation(te);
    PointCloudNT aligned;
    icp.align(aligned);

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "PCL:\n"
              << icp.getFinalTransformation() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClassPoint2Plane, Rotation6DOF)
{
    // Rotation
    // Eigen::Matrix3f rot; // Apparently this one is really tought. Not even PCL converges
    // rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
    //       Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
    //       Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;

    pcl::transformPointCloud(*target, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF6, PointXYZ, PointNormal>::dataset_t data;
    data.source = source;
    data.target = target_normals;
    data.tgt_search_method = kdtree_normals;

    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<DOF6, PointXYZ, PointNormal> *cost = new RegistrationCost<DOF6, PointXYZ, PointNormal>(&data);
    Registration<DOF6, PointXYZ, PointNormal> *registration = new Registration<DOF6, PointXYZ, PointNormal>(cost);

    registration->setMaxOptimizationIterations(1);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    // PCL ICP
    pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
    pcl::IterativeClosestPoint<PointNormal, PointNormal> icp; // PCL icp object is awkwardly templated
    PointCloudNT::Ptr source_normals(new PointCloudNT);
    pcl::copyPointCloud(*source, *source_normals);
    icp.setInputSource(source_normals);
    icp.setInputTarget(target_normals);
    icp.setMaxCorrespondenceDistance(MAXCORRDIST);
    icp.setMaximumIterations(MAXIT);
    icp.setSearchMethodTarget(kdtree_normals);
    pcl::registration::TransformationEstimationPointToPlane<PointNormal, PointNormal>::Ptr te(new pcl::registration::TransformationEstimationPointToPlane<PointNormal, PointNormal>);
    // pcl::registration::TransformationEstimationPointToPlaneLLS<PointNormal,PointNormal>::Ptr te
    // (new pcl::registration::TransformationEstimationPointToPlaneLLS<PointNormal,PointNormal>);
    icp.setTransformationEstimation(te);
    PointCloudNT aligned;
    icp.align(aligned);

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "PCL:\n"
              << icp.getFinalTransformation() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClassPoint2Plane, Rotation3DOF)
{
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(-0.9, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;

    pcl::transformPointCloud(*target, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF3, PointXYZ, PointNormal>::dataset_t data;
    data.source = source;
    data.target = target_normals;
    data.tgt_search_method = kdtree_normals;

    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<DOF3, PointXYZ, PointNormal> *cost = new RegistrationCost<DOF3, PointXYZ, PointNormal>(&data);
    Registration<DOF3, PointXYZ, PointNormal> *registration = new Registration<DOF3, PointXYZ, PointNormal>(cost);

    registration->setMaxOptimizationIterations(1);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

    Vector3N x0;
    x0.setZero();
    registration->minimize(x0);
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
    pcl::IterativeClosestPoint<PointNormal, PointNormal> icp; // PCL icp object is awkwardly templated
    PointCloudNT::Ptr source_normals(new PointCloudNT);
    pcl::copyPointCloud(*source, *source_normals);
    icp.setInputSource(source_normals);
    icp.setInputTarget(target_normals);
    icp.setMaxCorrespondenceDistance(MAXCORRDIST);
    icp.setMaximumIterations(MAXIT);
    icp.setSearchMethodTarget(kdtree_normals);
    pcl::registration::TransformationEstimationPointToPlane<PointNormal, PointNormal>::Ptr te(new pcl::registration::TransformationEstimationPointToPlane<PointNormal, PointNormal>);
    // pcl::registration::TransformationEstimationPointToPlaneLLS<PointNormal,PointNormal>::Ptr te
    // (new pcl::registration::TransformationEstimationPointToPlaneLLS<PointNormal,PointNormal>);
    icp.setTransformationEstimation(te);
    PointCloudNT aligned;
    icp.align(aligned);

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "PCL 6DOF:\n"
              << icp.getFinalTransformation() << std::endl;
    std::cerr << "Duna 3DOF:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClassPoint2Plane, RefEqualsGuess)
{
    // The reference rotation is a very tough case. So we HAVE to provide an initial guess to the solution.
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.9, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.9, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    pcl::transformPointCloud(*target, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF3, PointXYZ, PointNormal>::dataset_t data;
    data.source = source;
    data.target = target_normals;
    data.tgt_search_method = kdtree_normals;

    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<DOF3, PointXYZ, PointNormal> *cost = new RegistrationCost<DOF3, PointXYZ, PointNormal>(&data);
    Registration<DOF3, PointXYZ, PointNormal> *registration = new Registration<DOF3, PointXYZ, PointNormal>(cost);

    registration->setMaxOptimizationIterations(1);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

    // Our guess 
    // Vector3N x0(0.7,0.7,0.7);
    Eigen::Matrix4f reference_inverse = reference_transform.inverse();
    registration->minimize(reference_inverse);
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna 3DOF:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(RegistrationTestClassPoint2Plane, Guess3DOF)
{
    // The reference rotation is a very tough case. So we HAVE to provide an initial guess to the solution.
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.9, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.9, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    pcl::transformPointCloud(*target, *source, reference_transform);

    // Prepare dataset
    RegistrationCost<DOF3, PointXYZ, PointNormal>::dataset_t data;
    data.source = source;
    data.target = target_normals;
    data.tgt_search_method = kdtree_normals;

    // TODO ensure templated types are tightly coupled for dataset, cost and registration objects
    RegistrationCost<DOF3, PointXYZ, PointNormal> *cost = new RegistrationCost<DOF3, PointXYZ, PointNormal>(&data);
    Registration<DOF3, PointXYZ, PointNormal> *registration = new Registration<DOF3, PointXYZ, PointNormal>(cost);

    registration->setMaxOptimizationIterations(1);
    registration->setMaxIcpIterations(MAXIT);
    registration->setMaxCorrespondenceDistance(MAXCORRDIST);

    // Our guess 
    Vector3N x0(-0.5,-0.5,-0.5);
    registration->minimize(x0);
    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna 3DOF:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

// Extremmely useful for SLAM
TEST_F(RegistrationTestClassPoint2Plane, DISABLED_SeriesOfCalls3DOF)
{

    RegistrationCost<DOF3, PointXYZ, PointNormal>::dataset_t data;
    data.source = source;
    data.target = target_normals;
    data.tgt_search_method = kdtree_normals;

    RegistrationCost<DOF3, PointXYZ, PointNormal> *cost = new RegistrationCost<DOF3, PointXYZ, PointNormal>(&data);
    Registration<DOF3, PointXYZ, PointNormal> *registration = new Registration<DOF3, PointXYZ, PointNormal>(cost);

    for(int i=0; i < 10; ++ i)
    {



    }

   
}



