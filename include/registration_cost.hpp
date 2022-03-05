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
        m_source_transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        m_final_transform = Matrix4::Identity();

        // Checkdataset
        DUNA_DEBUG("source pts : %ld, tgt pts: %ld\n", l_dataset->source->size(),l_dataset->target->size());
    }
    virtual ~RegistrationCost() = default;

    inline void setMaxCorrDist(float dist)
    {
        m_max_correspondence_dist = dist;
    }

    void init(const VectorN &x0)
    {

        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        Matrix4 transform_matrix_;
        so3::param2Matrix(x0, transform_matrix_);

        // move point cloud
        pcl::transformPointCloud(*l_dataset->source, *m_source_transformed, transform_matrix_);
    }

    void preprocess(const VectorN &x0) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        m_correspondences.clear();
        m_correspondences.reserve(m_source_transformed->size());

        pcl::Indices indices(m_k_neighboors);
        std::vector<float> k_distances(m_k_neighboors);

        Eigen::Matrix4f transform_;
        so3::param2Matrix(x0, transform_);

        // compute correspondences
        for (int i = 0; i < m_source_transformed->size(); ++i)
        {

            const pcl::PointXYZ &pt_warped = m_source_transformed->points[i];

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

    void finalize(VectorN& x0) override
    {
        so3::matrix2Param(m_final_transform,x0);
        std::cerr << m_final_transform << std::endl;
        
    }

    void postprocess(VectorN &x0) override
    {
        Matrix4 transform_matrix_;
        so3::param2Matrix(x0, transform_matrix_);

        // move point cloud
        pcl::transformPointCloud(*m_source_transformed, *m_source_transformed, transform_matrix_);

        m_final_transform = transform_matrix_ *m_final_transform;        
        
        x0.setZero();
    }

    double computeCost(const VectorN &x) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);
        

        Matrix4 transform_;
        so3::param2Matrix(x, transform_);
        double sum = 0;
        for (int i = 0; i < m_correspondences.size(); ++i)
        {
            const pcl::PointXYZ &src_pt = m_source_transformed->points[m_correspondences[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[m_correspondences[i].index_match];

            const Eigen::Vector4f src_pt_vec = src_pt.getVector4fMap();
            const Eigen::Vector4f tgt_pt_vec = tgt_pt.getVector4fMap();
            const Eigen::Vector4f src_pt_warped_vec = transform_ * src_pt_vec;

            double xout = computeError(src_pt_warped_vec, tgt_pt_vec);

            sum += xout;
        }
        return sum;
    }

    double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        hessian.setZero();
        b.setZero();

        Eigen::Matrix<float, 1, NPARAM> jacobian_row;
        Eigen::Matrix4f transform_plus[NPARAM];
        Eigen::Matrix4f transform_minus[NPARAM];
        Eigen::Matrix4f transform_;

        so3::param2Matrix(x, transform_);

        const float epsilon = 1e-6;
        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            VectorN x_minus(x);

            x_plus[j] += epsilon;
            x_minus[j] -= epsilon;

            so3::param2Matrix(x_plus, transform_plus[j]);
            so3::param2Matrix(x_minus, transform_minus[j]);
        }

        double sum = 0;
        for (int i = 0; i < m_correspondences.size(); ++i)
        {
            const pcl::PointXYZ &src_pt = m_source_transformed->points[m_correspondences[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[m_correspondences[i].index_match];

            const Eigen::Vector4f src_pt_vec = src_pt.getVector4fMap();
            const Eigen::Vector4f tgt_pt_vec = tgt_pt.getVector4fMap();
            const Eigen::Vector4f src_pt_warped_vec = transform_ * src_pt_vec;

            double xout = computeError(src_pt_warped_vec, tgt_pt_vec);

            for (int j = 0; j < NPARAM; ++j)
            {
                Eigen::Vector4f src_pt_warped_plus_vec = transform_plus[j] * src_pt_vec;
                Eigen::Vector4f src_pt_warped_minus_vec = transform_minus[j] * src_pt_vec;

                double xout_plus = computeError(src_pt_warped_plus_vec, tgt_pt_vec);
                double xout_minus = computeError(src_pt_warped_minus_vec, tgt_pt_vec);

                jacobian_row[j] = (xout_plus - xout_minus) / (2 * epsilon);
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_source_transformed;
    Matrix4 m_final_transform;

    template <typename Scalar>
    inline double computeError(const Eigen::Matrix<Scalar, 3, 1> &src_vec, const Eigen::Matrix<Scalar, 3, 1> &tgt_vec)
    {
        return (src_vec.template cast<Scalar>() - tgt_vec.template cast<Scalar>()).norm();
    }

    template <typename Scalar>
    inline double computeError(const Eigen::Matrix<Scalar, 4, 1> &src_vec, const Eigen::Matrix<Scalar, 4, 1> &tgt_vec)
    {
        Eigen::Matrix<Scalar, 3, 1> src_vec3(src_vec[0], src_vec[1], src_vec[2]);
        Eigen::Matrix<Scalar, 3, 1> tgt_vec3(tgt_vec[0], tgt_vec[1], tgt_vec[2]);
        return (src_vec3 - tgt_vec3).norm();
    }
};