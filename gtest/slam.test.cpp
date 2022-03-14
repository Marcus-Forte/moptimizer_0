#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <duna/registration.h>

using namespace duna;
using PointT = pcl::PointNormal;
using PointCloudT = pcl::PointCloud<PointT>;

int main(int argc,char** argv){
    ::testing::InitGoogleTest(&argc,argv);



    return RUN_ALL_TESTS();
}




class SlamTest : public testing::Test {
public:
    SlamTest(){

    }

    virtual ~SlamTest(){}


    protected:
    PointCloudT::Ptr map;
    PointCloudT::Ptr source0;

};




TEST_F(SlamTest, SeriesOfScanMatchings){

}

