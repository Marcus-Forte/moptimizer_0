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
#include <duna/registration/transformation_estimation3DOF.h>
#include <duna/map/transformation_estimationMAP.h>
#include <duna/registration/scan_matching_3dof.h>
#include <pcl/registration/icp.h>

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
    }

protected:
    PointCloudT::Ptr target_;
    std::vector<PointCloudT::Ptr> source_vector_;
    pcl::search::KdTree<PointT>::Ptr target_kdtree_;
};

using ScalarTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(SequenceRegistration, ScalarTypes);

TYPED_TEST(SequenceRegistration, Indoor)
{

    std::cout << "Map size: " << this->target_->size() << std::endl;
    std::cout << "#Scans: " << this->source_vector_.size() << std::endl;

    pcl::IterativeClosestPoint<PointT, PointT, TypeParam> icp;

    // this is where the magic happens
    typename duna::TransformationEstimator3DOF<PointT, PointT, TypeParam>::Ptr duna_3dof_estimator(new duna::TransformationEstimator3DOF<PointT, PointT, TypeParam>(true));
    icp.setTransformationEstimation(duna_3dof_estimator);

    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(0.15);
    icp.setTransformationEpsilon(1e-6);
    // icp.setTransformationRotationEpsilon(1e-12);
    pcl::registration::CorrespondenceRejectorTrimmed::Ptr rejector0(new pcl::registration::CorrespondenceRejectorTrimmed);
    rejector0->setOverlapRatio(0.8);
    icp.addCorrespondenceRejector(rejector0);

    Eigen::Matrix<TypeParam, 4, 4> transform;
    transform.setIdentity();

    PointCloudT aligned;
    PointCloudT::Ptr subsampled_input(new PointCloudT);

    utilities::Stopwatch timer;
    PointCloudT::Ptr HD_cloud(new PointCloudT);

    // pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);

    pcl::NormalEstimation<PointT, PointT> ne;
    timer.tick();
    ne.setInputCloud(this->target_);
    ne.setSearchMethod(this->target_kdtree_);
    ne.setKSearch(15);
    ne.compute(*this->target_);
    timer.tock("Normal computation ");

    PointCloudT::Ptr output(new PointCloudT);

    // Copy full map cloud
    *HD_cloud = *this->target_;
    double total_reg_time = 0.0;
    for (int i = 0; i < this->source_vector_.size(); ++i)
    {
        std::cout << "Registering " << i << ": " << this->source_vector_[i]->size() << std::endl;

        pcl::UniformSampling<PointT> uniform_sampler;
        uniform_sampler.setInputCloud(this->target_);
        uniform_sampler.setRadiusSearch(0.1);
        uniform_sampler.filter(*this->target_);

        timer.tick();
        this->target_kdtree_->setInputCloud(this->target_);
        timer.tock("KDTree recomputation");

        // TODO how do we track new points only?

        timer.tick();
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(this->source_vector_[i]);
        voxel.setLeafSize(0.25, 0.25, 0.25);
        voxel.filter(*subsampled_input);
        timer.tock("Voxel grid.");

        std::cout << "Subsampled # points: " << subsampled_input->size() << std::endl;

        timer.tick();
        icp.setInputTarget(this->target_);
        icp.setSearchMethodTarget(this->target_kdtree_, true);
        icp.setInputSource(subsampled_input);
        icp.align(*output, transform);
        total_reg_time += timer.tock("Registration");

        timer.tick();

        // int new_points_size = output.size();
        // std::vector<int> new_point_indices(new_points_size);
        // std::iota(new_point_indices.begin(), new_point_indices.end(),this->target_->size() - new_points_size);
        // for (int l=0; l < new_points_size; ++ l)
        // {
        //     ASSERT_NEAR(this->target_->points[new_point_indices[l]].x, output[l].x, 1e-3);
        // }
        ne.setInputCloud(output);
        ne.setSearchSurface(this->target_);
        ne.setSearchMethod(this->target_kdtree_);
        ne.setKSearch(5);
        ne.compute(*output);
        *this->target_ += *output;

        timer.tock("Accumulation and normal recomputation");
        transform = icp.getFinalTransformation();

        pcl::transformPointCloud(*this->source_vector_[i], *output, transform);

        *HD_cloud = *HD_cloud + *output;
    }
    std::cout << "All registration took: " << total_reg_time << std::endl;

    PointCloudT::Ptr reference_map(new PointCloudT);
    pcl::io::loadPCDFile(TEST_DATA_DIR "/0_sequence_map_reference.pcd", *reference_map);

    double diff = this->compareClouds(reference_map, HD_cloud);

    std::cout << "Diff =  " << diff << std::endl;

    EXPECT_NEAR(diff, 0.0, 1e-2);

    // std::cout << "Filtering: " << HD_cloud->size() << std::endl;
    // pcl::RadiusOutlierRemoval<PointT> ror;
    // ror.setInputCloud(HD_cloud);
    // ror.setRadiusSearch(0.2);
    // ror.setMinNeighborsInRadius(500);
    // ror.filter(*HD_cloud);
    // std::cout << "Filtered to: " << HD_cloud->size() << std::endl;

    std::string final_cloud_filename = "sequence_3dof";
    if (std::is_same<TypeParam, float>::value)
        final_cloud_filename += "_float";
    else
        final_cloud_filename += "_double";

    final_cloud_filename += ".pcd";
    std::cout << "Saving final pointcloud to " << final_cloud_filename << std::endl;
    pcl::io::savePCDFileBinary(final_cloud_filename, *HD_cloud);
    pcl::io::savePCDFileBinary("ss_" + final_cloud_filename, *this->target_);
}

TYPED_TEST(SequenceRegistration, DunaScanMatcherIndoor)
{

    std::cout << "Map size: " << this->target_->size() << std::endl;
    std::cout << "#Scans: " << this->source_vector_.size() << std::endl;

    typename duna::ScanMatching3DOF<PointT, PointT, TypeParam>::Ptr scan_matching(new duna::ScanMatching3DOF<PointT, PointT, TypeParam>);
    scan_matching->getLogger().setVerbosityLevel(duna::L_DEBUG);
    // icp.setTransformationEstimation(duna_3dof_estimator);

    // icp.setMaximumIterations(50);
    // icp.setMaxCorrespondenceDistance(0.15);
    // icp.setTransformationEpsilon(1e-6);
    // // icp.setTransformationRotationEpsilon(1e-12);
    // pcl::registration::CorrespondenceRejectorTrimmed::Ptr rejector0(new pcl::registration::CorrespondenceRejectorTrimmed);
    // rejector0->setOverlapRatio(0.8);
    // icp.addCorrespondenceRejector(rejector0);

    Eigen::Matrix<TypeParam, 4, 4> transform;
    transform.setIdentity();

    PointCloudT aligned;
    PointCloudT::Ptr subsampled_input(new PointCloudT);

    utilities::Stopwatch timer;
    PointCloudT::Ptr HD_cloud(new PointCloudT);

    // pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);

    pcl::NormalEstimation<PointT, PointT> ne;
    timer.tick();
    ne.setInputCloud(this->target_);
    ne.setSearchMethod(this->target_kdtree_);
    ne.setKSearch(15);
    ne.compute(*this->target_);
    timer.tock("Normal computation ");

    PointCloudT::Ptr output(new PointCloudT);

    // Copy full map cloud
    *HD_cloud = *this->target_;
    double total_reg_time = 0.0;
    for (int i = 0; i < this->source_vector_.size(); ++i)
    {
        std::cout << "Registering " << i << ": " << this->source_vector_[i]->size() << std::endl;

        pcl::UniformSampling<PointT> uniform_sampler;
        uniform_sampler.setInputCloud(this->target_);
        uniform_sampler.setRadiusSearch(0.1);
        uniform_sampler.filter(*this->target_);

        timer.tick();
        this->target_kdtree_->setInputCloud(this->target_);
        timer.tock("KDTree recomputation");

        // TODO how do we track new points only?

        timer.tick();
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(this->source_vector_[i]);
        voxel.setLeafSize(0.25, 0.25, 0.25);
        voxel.filter(*subsampled_input);
        timer.tock("Voxel grid.");

        std::cout << "Subsampled # points: " << subsampled_input->size() << std::endl;

        timer.tick();

        scan_matching->setMaxCorrDistance(0.8);
        scan_matching->setMaxNumIterations(50);
        scan_matching->setMaxNumOptIterations(10);
        scan_matching->setInputTarget(this->target_);
        scan_matching->setTargetSearchTree(this->target_kdtree_);
        scan_matching->setInputSource(subsampled_input);
        scan_matching->match(transform);

        total_reg_time += timer.tock("Scan matching");

        timer.tick();

        // int new_points_size = output.size();
        // std::vector<int> new_point_indices(new_points_size);
        // std::iota(new_point_indices.begin(), new_point_indices.end(),this->target_->size() - new_points_size);
        // for (int l=0; l < new_points_size; ++ l)
        // {
        //     ASSERT_NEAR(this->target_->points[new_point_indices[l]].x, output[l].x, 1e-3);
        // }
        ne.setInputCloud(output);
        ne.setSearchSurface(this->target_);
        ne.setSearchMethod(this->target_kdtree_);
        ne.setKSearch(5);
        ne.compute(*output);
        *this->target_ += *output;

        timer.tock("Accumulation and normal recomputation");
        transform = scan_matching->getFinalTransformation();

        pcl::transformPointCloud(*this->source_vector_[i], *output, transform);

        *HD_cloud = *HD_cloud + *output;
    }
    std::cout << "All registration took: " << total_reg_time << std::endl;

    PointCloudT::Ptr reference_map(new PointCloudT);
    pcl::io::loadPCDFile(TEST_DATA_DIR "/0_sequence_map_reference.pcd", *reference_map);

    double diff = this->compareClouds(reference_map, HD_cloud);

    std::cout << "Diff =  " << diff << std::endl;

    EXPECT_NEAR(diff, 0.0, 1e-2);

    // std::cout << "Filtering: " << HD_cloud->size() << std::endl;
    // pcl::RadiusOutlierRemoval<PointT> ror;
    // ror.setInputCloud(HD_cloud);
    // ror.setRadiusSearch(0.2);
    // ror.setMinNeighborsInRadius(500);
    // ror.filter(*HD_cloud);
    // std::cout << "Filtered to: " << HD_cloud->size() << std::endl;

    std::string final_cloud_filename = "sequence_3dof_matcher";
    if (std::is_same<TypeParam, float>::value)
        final_cloud_filename += "_float";
    else
        final_cloud_filename += "_double";

    final_cloud_filename += ".pcd";
    std::cout << "Saving final pointcloud to " << final_cloud_filename << std::endl;
    pcl::io::savePCDFileBinary(final_cloud_filename, *HD_cloud);
    pcl::io::savePCDFileBinary("ss_" + final_cloud_filename, *this->target_);
}

TYPED_TEST(SequenceRegistration, DISABLED_IndoorMAP)
{

    std::cout << "Map size: " << this->target_->size() << std::endl;
    std::cout << "#Scans: " << this->source_vector_.size() << std::endl;

    pcl::IterativeClosestPoint<PointT, PointT, TypeParam> icp;

    // this is where the magic happens
    typename duna::TransformationEstimatorMAP<PointT, PointT, TypeParam>::Ptr duna_3dof_estimator(new duna::TransformationEstimatorMAP<PointT, PointT, TypeParam>(true));
    icp.setTransformationEstimation(duna_3dof_estimator);

    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(0.15);
    icp.setTransformationEpsilon(1e-6);
    // icp.setTransformationRotationEpsilon(1e-12);
    pcl::registration::CorrespondenceRejectorTrimmed::Ptr rejector0(new pcl::registration::CorrespondenceRejectorTrimmed);
    rejector0->setOverlapRatio(0.8);
    icp.addCorrespondenceRejector(rejector0);

    Eigen::Matrix<TypeParam, 4, 4> transform;
    transform.setIdentity();

    PointCloudT aligned;
    PointCloudT::Ptr subsampled_input(new PointCloudT);

    utilities::Stopwatch timer;
    PointCloudT::Ptr HD_cloud(new PointCloudT);

    // pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);

    pcl::NormalEstimation<PointT, PointT> ne;
    timer.tick();
    ne.setInputCloud(this->target_);
    ne.setSearchMethod(this->target_kdtree_);
    ne.setKSearch(15);
    ne.compute(*this->target_);
    timer.tock("Normal computation ");

    PointCloudT::Ptr output(new PointCloudT);

    // Copy full map cloud
    *HD_cloud = *this->target_;
    double total_reg_time = 0.0;
    for (int i = 0; i < this->source_vector_.size(); ++i)
    {
        std::cout << "Registering " << i << ": " << this->source_vector_[i]->size() << std::endl;

        pcl::UniformSampling<PointT> uniform_sampler;
        uniform_sampler.setInputCloud(this->target_);
        uniform_sampler.setRadiusSearch(0.1);
        uniform_sampler.filter(*this->target_);

        timer.tick();
        this->target_kdtree_->setInputCloud(this->target_);
        timer.tock("KDTree recomputation");

        // TODO how do we track new points only?

        timer.tick();
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(this->source_vector_[i]);
        voxel.setLeafSize(0.25, 0.25, 0.25);
        voxel.filter(*subsampled_input);
        timer.tock("Voxel grid.");

        std::cout << "Subsampled # points: " << subsampled_input->size() << std::endl;

        timer.tick();
        icp.setInputTarget(this->target_);
        icp.setSearchMethodTarget(this->target_kdtree_);
        icp.setInputSource(subsampled_input);
        icp.align(*output, transform);
        total_reg_time += timer.tock("Registration");

        timer.tick();

        // int new_points_size = output.size();
        // std::vector<int> new_point_indices(new_points_size);
        // std::iota(new_point_indices.begin(), new_point_indices.end(),this->target_->size() - new_points_size);
        // for (int l=0; l < new_points_size; ++ l)
        // {
        //     ASSERT_NEAR(this->target_->points[new_point_indices[l]].x, output[l].x, 1e-3);
        // }
        ne.setInputCloud(output);
        ne.setSearchSurface(this->target_);
        ne.setSearchMethod(this->target_kdtree_);
        ne.setKSearch(5);
        ne.compute(*output);
        *this->target_ += *output;

        timer.tock("Accumulation and normal recomputation");
        transform = icp.getFinalTransformation();

        pcl::transformPointCloud(*this->source_vector_[i], *output, transform);

        *HD_cloud = *HD_cloud + *output;
    }
    std::cout << "All registration took: " << total_reg_time << std::endl;

    PointCloudT::Ptr reference_map(new PointCloudT);
    pcl::io::loadPCDFile(TEST_DATA_DIR "/0_sequence_map_reference.pcd", *reference_map);

    double diff = this->compareClouds(reference_map, HD_cloud);

    std::cout << "Diff =  " << diff << std::endl;

    EXPECT_NEAR(diff, 0.0, 1e-2);

    // std::cout << "Filtering: " << HD_cloud->size() << std::endl;
    // pcl::RadiusOutlierRemoval<PointT> ror;
    // ror.setInputCloud(HD_cloud);
    // ror.setRadiusSearch(0.2);
    // ror.setMinNeighborsInRadius(500);
    // ror.filter(*HD_cloud);
    // std::cout << "Filtered to: " << HD_cloud->size() << std::endl;

    std::string final_cloud_filename = "sequence_3dof";
    if (std::is_same<TypeParam, float>::value)
        final_cloud_filename += "_float";
    else
        final_cloud_filename += "_double";

    final_cloud_filename += ".pcd";
    std::cout << "Saving final pointcloud to " << final_cloud_filename << std::endl;
    pcl::io::savePCDFileBinary(final_cloud_filename, *HD_cloud);
    pcl::io::savePCDFileBinary("ss_" + final_cloud_filename, *this->target_);
}