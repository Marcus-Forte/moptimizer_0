#pragma once

#include "cost_function.hpp"
#include "so3.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>

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
 
    using Matrix4 = Eigen::Matrix4f;

    using CostFunction<NPARAM>::m_dataset;
   

    RegistrationCost(void *dataset) : CostFunction<NPARAM>(dataset)
    {
        l_dataset = reinterpret_cast<datatype_t *>(m_dataset);
        // Checkdataset
        DUNA_DEBUG("source pts : %ld, tgt pts: %ld\n", l_dataset->source->size(),l_dataset->target->size());
        // TODO check KDTREE
    }

    // TODO check numerical stability
    virtual ~RegistrationCost() = default;


    // TODO implement
    void checkData() override 
    {
        
    }

    inline void setTransformedSourcePtr(pcl::PointCloud<pcl::PointXYZ>::ConstPtr transformed_src ){
        m_transformed_source = transformed_src;
    }

    inline void setCorrespondencesPtr(pcl::CorrespondencesConstPtr correspondences){
        m_correspondences = correspondences;
    }



    double computeCost(const VectorN &x) override
    {

        Matrix4 transform_;
        so3::param2Matrix6DOF(x, transform_);
        double sum = 0;
        for (int i = 0; i < m_correspondences->size(); ++i)
        {
            const pcl::PointXYZ &src_pt = m_transformed_source->points[(*m_correspondences)[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[(*m_correspondences)[i].index_match];

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

        hessian.setZero();
        b.setZero();

        Eigen::Matrix<float, 1, NPARAM> jacobian_row;
        Eigen::Matrix4f transform_plus[NPARAM];
        Eigen::Matrix4f transform_minus[NPARAM];
        Eigen::Matrix4f transform_;

        so3::param2Matrix6DOF(x, transform_);

        const float epsilon = 1e-6;
        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            VectorN x_minus(x);

            x_plus[j] += epsilon;
            x_minus[j] -= epsilon;

            so3::param2Matrix6DOF(x_plus, transform_plus[j]);
            so3::param2Matrix6DOF(x_minus, transform_minus[j]);
        }

        double sum = 0;
        for (int i = 0; i < m_correspondences->size(); ++i)
        {
            const pcl::PointXYZ &src_pt = m_transformed_source->points[(*m_correspondences)[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[(*m_correspondences)[i].index_match];

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
    pcl::CorrespondencesConstPtr m_correspondences;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr m_transformed_source;
    datatype_t *l_dataset; // cast
    

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