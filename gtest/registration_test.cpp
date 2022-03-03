#include "registration.hpp"
#include "registration_cost.hpp"
#include "duna_log.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <gtest/gtest.h>

#ifndef TEST_DATA_DIR
#warning "NO 'TEST_DATA_DIR' DEFINED"
#define TEST_DATA_DIR "./"
#endif

#define MODEL_PARAM 6
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
    
    referece_transform.topLeftCorner<3,3>() = rot;
    // Translation
    referece_transform.col(3) = Eigen::Vector4f(1,2,3,1);

    pcl::transformPointCloud(*target,*source,referece_transform);

    Eigen::Vector4f target_centroid;
    pcl::compute3DCentroid(*target,target_centroid);

    Eigen::Vector4f source_centroid;
    pcl::compute3DCentroid(*source,source_centroid);

    for(int i=0; i < 3; ++i){
        ASSERT_NE(target_centroid[i], source_centroid[i]);
    }

    

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(target);

    // Prepare dataset
    data.source = source;
    data.target = target;
    data.tgt_kdtree = kdtree;
    cost = new RegistrationCost<MODEL_PARAM>(target->size(),&data);
    registration = new Registration<MODEL_PARAM>(cost);

    cost->setMaxCorrDist(15);

    VectorN x0;
    x0.setZero();
    registration->minimize(x0);


    exit(-1);
    
}
