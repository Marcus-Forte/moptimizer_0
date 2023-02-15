#define PCL_NO_PRECOMPILE
#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/features/normal_3d.h>

#include <duna/stopwatch.hpp>
#include <duna/models/scan_matching.h>
#include <duna/cost_function_numerical.h>
#include <duna/levenberg_marquadt.h>
#include <duna/loss_function/geman_mcclure.h>

#define N_MAP 20
#define N_SOURCE 85
#define N_SOURCE_MERGE 1

using PointT = pcl::PointXYZINormal;
using PointCloudT = pcl::PointCloud<PointT>;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename Scalar>
class SequenceRegistration : public ::testing::Test
{
protected:
    double compareClouds(const PointCloudT::ConstPtr &cloud_a, const PointCloudT::ConstPtr &cloud_b)
    {
        double diff = 0;
        // Find correspondences
        pcl::search::KdTree<PointT> cloud_a_kdtree;
        cloud_a_kdtree.setInputCloud(cloud_a);

        std::vector<int> indices;
        std::vector<float> unused;
        for (const auto &it : cloud_b->points)
        {
            cloud_a_kdtree.nearestKSearch(it, 1, indices, unused);
            diff += (cloud_a->points[indices[0]].getVector3fMap() - it.getVector3fMap()).norm();
        }

        // RMSE
        return diff / (double)cloud_b->size();
    }

public:
    SequenceRegistration()
    {
        std::string map_filename = "/map";
        std::string scan_filename = "/scan";
        PointCloudT::Ptr temp(new PointCloudT);

        target_.reset(new PointCloudT);

        for (int i = 1; i < N_MAP; ++i)
        {
            std::cerr << "Map Loading: " << map_filename + std::to_string(i) << std::endl;
            if (pcl::io::loadPCDFile(TEST_DATA_DIR + map_filename + std::to_string(i) + ".pcd", *temp) != 0)
            {
                std::cerr << "Unable to load dataset\n";
                exit(-1);
            }

            *target_ = *target_ + *temp;
        }

        // Remove points clode to origin
        pcl::PassThrough<PointT> passthrough;
        passthrough.setInputCloud(target_);
        passthrough.setFilterLimits(.1, 100);
        passthrough.setFilterFieldName("x");
        passthrough.filter(*target_);

        PointCloudT::Ptr source_merge(new PointCloudT);

        for (int i = 1; i < N_SOURCE; ++i)
        {
            PointCloudT::Ptr temp_scan(new PointCloudT);
            std::cerr << "Scan Loading: " << scan_filename + std::to_string(i) << std::endl;
            if (pcl::io::loadPCDFile(TEST_DATA_DIR + scan_filename + std::to_string(i) + ".pcd", *temp_scan) != 0)
            {
                std::cerr << "Unable to load dataset\n";
                exit(-1);
            }

            *source_merge += *temp_scan;

            if (i % N_SOURCE_MERGE == 0)
            {
                source_vector_.push_back(source_merge);
                source_merge.reset(new PointCloudT);
            }
        }
        target_kdtree_.reset(new pcl::search::KdTree<PointT>);
        subsample = 0.1;
    }

protected:
    PointCloudT::Ptr target_;
    std::vector<PointCloudT::Ptr> source_vector_;
    pcl::search::KdTree<PointT>::Ptr target_kdtree_;
    float subsample;
};

using ScalarTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(SequenceRegistration, ScalarTypes);

TYPED_TEST(SequenceRegistration, OptimizerIndoor)
{

    std::cout << "Map size: " << this->target_->size() << std::endl;
    std::cout << "#Scans: " << this->source_vector_.size() << std::endl;

    duna::LevenbergMarquadt<TypeParam, 3> optimizer;
    optimizer.setMaximumIterations(15);

    Eigen::Matrix<TypeParam, 4, 4> transform;
    transform.setIdentity();

    PointCloudT aligned;
    PointCloudT::Ptr subsampled_input(new PointCloudT);

    utilities::Stopwatch timer;
    PointCloudT::Ptr HD_cloud(new PointCloudT);

    pcl::NormalEstimation<PointT, PointT> ne;
    timer.tick();
    ne.setInputCloud(this->target_);
    ne.setSearchMethod(this->target_kdtree_);
    ne.setKSearch(15);
    ne.compute(*this->target_);
    timer.tock("Normal computation ");

    PointCloudT::Ptr output(new PointCloudT);

    duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);

    pcl::registration::CorrespondenceRejectorTrimmed::Ptr rejector0(new pcl::registration::CorrespondenceRejectorTrimmed);
    rejector0->setOverlapRatio(0.8);

    // Copy full map cloud
    *HD_cloud = *this->target_;
    double total_reg_time = 0.0;
    TypeParam x0[3];
    x0[0] = 0;
    x0[1] = 0;
    x0[2] = 0;
    typename duna::ScanMatchingBase<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
    for (int i = 0; i < this->source_vector_.size(); ++i)
    // for (int i = 0; i < 2; ++i)
    {
        std::cout << "Registering " << i << ": " << this->source_vector_[i]->size() << std::endl;

        pcl::UniformSampling<PointT> uniform_sampler;
        uniform_sampler.setInputCloud(this->target_);
        uniform_sampler.setRadiusSearch(0.1);
        uniform_sampler.filter(*this->target_);

        timer.tick();
        this->target_kdtree_->setInputCloud(this->target_);
        timer.tock("KDTree recomputation");

        timer.tick();
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(this->source_vector_[i]);
        voxel.setLeafSize(this->subsample, this->subsample, this->subsample);
        voxel.filter(*subsampled_input);
        timer.tock("Voxel grid.");

        std::cout << "Subsampled # points: " << subsampled_input->size() << std::endl;

        timer.tick();

        scan_matcher_model.reset(new duna::ScanMatching3DOFPoint2Plane<PointT, PointT, TypeParam>(subsampled_input, this->target_, this->target_kdtree_));
        // scan_matcher_model.reset(new duna::ScanMatching3DOFPoint2Point<PointT, PointT, TypeParam>(subsampled_input, this->target_, this->target_kdtree_));
        scan_matcher_model->setMaximumCorrespondenceDistance(0.15);
        scan_matcher_model->addCorrespondenceRejector(rejector0);
        auto cost = new duna::CostFunctionNumericalDiff<TypeParam, 3, 3>(scan_matcher_model, subsampled_input->size());
        cost->setLossFunction(typename duna::loss::GemmanMCClure<TypeParam>::Ptr(new duna::loss::GemmanMCClure<TypeParam>(0.1)));
        
        optimizer.addCost(cost);
        optimizer.minimize(x0);
        optimizer.clearCosts();
        total_reg_time += timer.tock("Registration");
        delete cost;

        // std::cout << "x = " << x0[0] << "," << x0[1] << "," << x0[2] << std::endl;

        so3::convert3DOFParameterToMatrix(x0, transform);
        pcl::transformPointCloud(*subsampled_input, *output, transform);

        timer.tick();

        ne.setInputCloud(output);
        ne.setSearchSurface(this->target_);
        ne.setSearchMethod(this->target_kdtree_);
        ne.setKSearch(5);
        ne.compute(*output);
        *this->target_ += *output;

        timer.tock("Accumulation and normal recomputation");

        pcl::transformPointCloud(*this->source_vector_[i], *output, transform);

        *HD_cloud = *HD_cloud + *output;
    }
    std::cout << "All registration took: " << total_reg_time << std::endl;

    PointCloudT::Ptr reference_map(new PointCloudT);
    pcl::io::loadPCDFile(TEST_DATA_DIR "/0_sequence_map_reference.pcd", *reference_map);

    double diff = this->compareClouds(reference_map, HD_cloud);

    std::cout << "Diff =  " << diff << std::endl;

    EXPECT_NEAR(diff, 0.0, 1e-2);

    std::string final_cloud_filename = "sequence_3dof_pure_optim";
    if (std::is_same<TypeParam, float>::value)
        final_cloud_filename += "_float";
    else
        final_cloud_filename += "_double";

    final_cloud_filename += ".pcd";
    std::cout << "Saving final pointcloud to " << final_cloud_filename << std::endl;
    pcl::io::savePCDFileBinary(final_cloud_filename, *HD_cloud);
    pcl::io::savePCDFileBinary("ss_" + final_cloud_filename, *this->target_);
}