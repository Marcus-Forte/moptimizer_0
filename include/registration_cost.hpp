#pragma once

#include "cost_function.hpp"
#include "so3.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

#include <assert.h>
/*

/* Define your dataset */
struct datatype_t
{
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr source;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr target;
    pcl::search::KdTree<pcl::PointXYZ>::ConstPtr tgt_kdtree;
};

template <int NPARAM>
class RegistrationCost : public CostFunction<NPARAM>
{
public:
    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using MatrixN = typename CostFunction<NPARAM>::MatrixN;
    using VectorX = typename CostFunction<NPARAM>::VectorX;
    using MatrixX = typename CostFunction<NPARAM>::MatrixX;
    using Matrix4 = Eigen::Matrix4f;

    using CostFunction<NPARAM>::m_dataset;
    using CostFunction<NPARAM>::m_data_size;

    RegistrationCost(unsigned int data_size, void *dataset) : CostFunction<NPARAM>(data_size, dataset)
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);
        // m_source_transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    virtual ~RegistrationCost() = default;

    inline void setMaxCorrDist(float dist)
    {
        m_max_correspondence_dist = dist;
    }

    // void init(const VectorN &x0)
    // {

    //     datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

    //     // Matrix4 transform_matrix_;
    //     // so3::param2Matrix(x0, transform_matrix_);

    //     // move point cloud
    //     // pcl::transformPointCloud(*l_dataset->source, *m_source_transformed, transform_matrix_);
    // }

    void preprocess(const VectorN &x0) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        m_correspondences.clear();
        m_correspondences.reserve(l_dataset->source->size());

        pcl::Indices indices(m_k_neighboors);
        std::vector<float> k_distances(m_k_neighboors);

        Eigen::Matrix4f transform_;
        so3::param2Matrix(x0,transform_);

        // compute correspondences
        for (int i = 0; i < l_dataset->source->size(); ++i)
        {
            
            pcl::PointXYZ pt_warped;
            pt_warped.getVector4fMap() = transform_ * l_dataset->source->points[i].getVector4fMap();

            l_dataset->tgt_kdtree->nearestKSearch(pt_warped, m_k_neighboors, indices, k_distances);

            if (k_distances[0] > m_max_correspondence_dist * m_max_correspondence_dist)
                continue;

            // Compute normal

            pcl::Correspondence correspondence;
            correspondence.index_match = indices[0];
            correspondence.index_query = i;
            m_correspondences.push_back(correspondence);
        }

        DUNA_DEBUG("source pts : %ld, corr pts: %ld\n", l_dataset->source->size(), m_correspondences.size());

        if (m_correspondences.size() == 0)
        {
            throw std::runtime_error("no more correspondences.");
        }
    }

    // void postprocess(const VectorN &x0) override
    // {
    //     Matrix4 transform_matrix_;
    //     so3::param2Matrix(x0, transform_matrix_);

    //     // move point cloud
    //     pcl::transformPointCloud(*m_source_transformed, *m_source_transformed, transform_matrix_);
    // }

    double computeCost(const VectorN &x) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);
        double sum = 0;

        Matrix4 transform_;
        so3::param2Matrix(x, transform_);
        
        //  DUNA_DEBUG_STREAM("x\n" << x << "\n");
        // DUNA_DEBUG_STREAM("transform\n" << transform_ << "\n");

        for (int i = 0; i < m_correspondences.size(); ++i)
        {
            const pcl::PointXYZ &src_pt = l_dataset->source->points[m_correspondences[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[m_correspondences[i].index_match];

            const Eigen::Vector4f &src_pt_vec = src_pt.getVector4fMap();
            const Eigen::Vector4f &tgt_pt_vec = tgt_pt.getVector4fMap();

            double xout = computeError(src_pt_vec, tgt_pt_vec, transform_);

            sum += xout;
        }

        return sum;
    }

    double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        double sum = 0;
    
        hessian.setZero();
        b.setZero();

        // Build incremental transformations
        Eigen::Matrix4f transform_plus[NPARAM];
        // Eigen::Matrix4f transform_minus[NPARAM];
        Eigen::Matrix4f transform_;

        so3::param2Matrix(x, transform_);

        Eigen::Matrix<float, 1, NPARAM> jacobian_row;

        const double epsilon = 0.0000001;
        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            x_plus[j] += epsilon;

            so3::param2Matrix(x_plus, transform_plus[j]);
        }

        
        for (int i = 0; i < m_correspondences.size(); ++i)
        {
            const pcl::PointXYZ &src_pt = l_dataset->source->points[m_correspondences[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[m_correspondences[i].index_match];

            const Eigen::Vector4f &src_pt_vec = src_pt.getVector4fMap();
            const Eigen::Vector4f &tgt_pt_vec = tgt_pt.getVector4fMap();

            // DUNA_DEBUG_STREAM("src[" << m_correspondences[i].index_query << "] " 
            //                 "<--> tgt[" << m_correspondences[i].index_match << "]\n"
            //                  << src_pt_vec << "-\n"  << tgt_pt_vec << std::endl);

            double xout = computeError(src_pt_vec, tgt_pt_vec, transform_);

    
            for (int j = 0; j < NPARAM; ++j)
            {
                double xout_plus = computeError(src_pt_vec, tgt_pt_vec, transform_plus[j]);

                jacobian_row[j] = (xout_plus - xout) / epsilon;
            }

            hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
            b += jacobian_row.transpose() * xout;

            sum += xout;
        }

         hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

        return sum;
    }

private:
    float m_max_correspondence_dist = 1.0;
    unsigned int m_k_neighboors = 5;
    pcl::Correspondences m_correspondences;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr m_source_transformed;

    double computeError(const Eigen::Vector4f &src_vec, const Eigen::Vector4f &tgt_vec, const Eigen::Matrix4f &transform_)
    {
        Eigen::Vector4f src_pt_vec_warped = transform_ * src_vec;

        Eigen::Vector3f src(src_pt_vec_warped[0],src_pt_vec_warped[1],src_pt_vec_warped[2]);
        Eigen::Vector3f tgt(tgt_vec[0],tgt_vec[1],tgt_vec[2]);
       

        return (src-tgt).norm();
    }
};