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
// Optimization objects
Registration<MODEL_PARAM>* registration;
RegistrationCost<MODEL_PARAM>* cost;
datatype_t data;




using PointCloudT = pcl::PointCloud<pcl::PointXYZ>;
using VectorN = CostFunction<MODEL_PARAM>::VectorN;
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i)
    {
        printf("arg: %2d = %s\n", i, argv[i]);
    }  




 
    

    return RUN_ALL_TESTS();
}

TEST(RegistrationTest, Test0){

    PointCloudT::Ptr source (new PointCloudT);
    PointCloudT::Ptr target (new PointCloudT);

    if( pcl::io::loadPCDFile(TEST_DATA_DIR,*target) != 0){
       std::cerr << "Make sure you run the rest at the binaries folder.\n";
       FAIL();
    }

    std::cout << "Loaded " << target->points.size() << " points\n";



    Eigen::MatrixX4f referece_transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.25 , Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.15 , Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.45 , Eigen::Vector3f::UnitZ());
    
    // referece_transform.topLeftCorner<3,3>() = rot;
    // Translation
    referece_transform.col(3) = Eigen::Vector4f(0.5,0.5,0.5,1);

   

    pcl::transformPointCloud(*target,*source,referece_transform);

    Eigen::Vector4f target_centroid;
    pcl::compute3DCentroid(*target,target_centroid);

    Eigen::Vector4f source_centroid;
    pcl::compute3DCentroid(*source,source_centroid);

    // for(int i=0; i < 3; ++i){
    //     ASSERT_NE(target_centroid[i], source_centroid[i]);
    // }

    

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(target);

    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(1.0);
    icp.setMaximumIterations(MAXIT);
    pcl::registration::TransformationEstimationLM<pcl::PointXYZ,pcl::PointXYZ>::Ptr lm_est(new pcl::registration::TransformationEstimationLM<pcl::PointXYZ,pcl::PointXYZ>);
    icp.setTransformationEstimation(lm_est);
    

    pcl::PointCloud<pcl::PointXYZ> aligned;

    icp.align(aligned);

    Eigen::Matrix4f final_reg_pcl = icp.getFinalTransformation();
    
    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;
    cost = new RegistrationCost<MODEL_PARAM>(target->size(),&data);
    registration = new Registration<MODEL_PARAM>(cost);

    registration->setMaxIt(MAXIT);

    cost->setMaxCorrDist(1);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);

    Eigen::Matrix4f final_reg_duna;
    so3::param2Matrix(x0,final_reg_duna);

    std::cerr << "Reference:\n" << referece_transform << std::endl;
    std::cerr << "PCL:\n" << final_reg_pcl << std::endl;
    std::cerr << "Duna:\n" << final_reg_duna << std::endl;

    for(int i=0;i <final_reg_pcl.size();i++ ){
        EXPECT_NEAR(final_reg_pcl.data()[i], final_reg_duna.data()[i], 0.01);
    }


    exit(-1);
    
}
