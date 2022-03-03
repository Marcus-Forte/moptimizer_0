#pragma once

#include "cost_function.hpp"
#include <assert.h>

#include <Eigen/Dense>
#include <vector>
#include <pcl/point_types.h>

/* Define your dataset */
struct camera_calibration_data_t
{
    camera_calibration_data_t()
    {
        camera_lidar_frame.setIdentity();
        CameraModel.setIdentity();
        pixel_list.clear();
        point_list.clear();
    }
    struct pixel_pair
    {
        pixel_pair(int x_, int y_) : x(x_), y(y_) {}
        int x;
        int y;
    };

    std::vector<pixel_pair> pixel_list;
    std::vector<pcl::PointXYZ> point_list;
    Eigen::Matrix<double, 3, 4> CameraModel;
    Eigen::Matrix4d camera_lidar_frame;
};

template <int NPARAM>
class CalibrationCost : public CostFunction<NPARAM>
{
public:
    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using MatrixN = typename CostFunction<NPARAM>::MatrixN;
    using VectorX = typename CostFunction<NPARAM>::VectorX;
    using MatrixX = typename CostFunction<NPARAM>::MatrixX;

    using CostFunction<NPARAM>::m_dataset;
    // using CostFunction<NPARAM>::m_data_size;

    CalibrationCost(unsigned int data_size, void *dataset) : CostFunction<NPARAM>(data_size, dataset) {}
    virtual ~CalibrationCost() = default;

    double computeCost(const VectorN &x) override
    {
        camera_calibration_data_t *l_dataset = reinterpret_cast<camera_calibration_data_t *>(m_dataset);
        double sum = 0;

        Eigen::Matrix4d transform = param2Matrix(x);

        for (int i = 0; i < l_dataset->point_list.size(); ++i)
        {

            Eigen::Vector3d out_pixel;
            out_pixel = l_dataset->CameraModel * transform * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast <double>();

            // compose error vector
            Eigen::Vector2d xout;
            xout[0] = l_dataset->pixel_list[i].x - (out_pixel[0] / out_pixel[2]);
            xout[1] = l_dataset->pixel_list[i].y - (out_pixel[1] / out_pixel[2]);

            sum += xout.squaredNorm();
        }
        return sum;
    }

    double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) override
    {

        camera_calibration_data_t *l_dataset = reinterpret_cast<camera_calibration_data_t *>(m_dataset);

        double sum = 0;
        hessian.setZero();
        b.setZero();

        // Build matrix from xi
        Eigen::Matrix4d transform = param2Matrix(x);

        Eigen::Matrix<double, 2, NPARAM> jacobian_row;

        // Build incremental transformations
        Eigen::Matrix4d transform_plus[NPARAM];
        Eigen::Matrix4d transform_minus[NPARAM];

        Eigen::Matrix<double,NPARAM,NPARAM> hessian_;
        Eigen::Matrix<double,NPARAM,1> b_;
        hessian_.setZero();
        b_.setZero();

        const double epsilon = 0.00001;
        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            VectorN x_minus(x);
            x_plus[j] += epsilon;
            x_minus[j] -= epsilon;

            transform_plus[j] = param2Matrix(x_plus);
            transform_minus[j] = param2Matrix(x_minus);
        }

        for (int i = 0; i < l_dataset->point_list.size(); ++i)
        {

            Eigen::Vector3d out_pixel;
            out_pixel = l_dataset->CameraModel * transform * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast <double>();

            // compose error vector
            Eigen::Vector2d xout;
            xout[0] = l_dataset->pixel_list[i].x - (out_pixel[0] / out_pixel[2]);
            xout[1] = l_dataset->pixel_list[i].y - (out_pixel[1] / out_pixel[2]);

            for (int j = 0; j < NPARAM; ++j)
            {

                Eigen::Vector3d out_pixel_plus, out_pixel_minus;
                out_pixel_plus = l_dataset->CameraModel * transform_plus[j] * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast <double>();
                out_pixel_minus = l_dataset->CameraModel * transform_minus[j] * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast <double>();

                Eigen::Vector2d xout_plus, xout_minus;
                xout_plus[0] = l_dataset->pixel_list[i].x - (out_pixel_plus[0] / out_pixel_plus[2]);
                xout_plus[1] = l_dataset->pixel_list[i].y - (out_pixel_plus[1] / out_pixel_plus[2]);

                xout_minus[0] = l_dataset->pixel_list[i].x - (out_pixel_minus[0] / out_pixel_minus[2]);
                xout_minus[1] = l_dataset->pixel_list[i].y - (out_pixel_minus[1] / out_pixel_minus[2]);

                jacobian_row.col(j) = (xout_plus - xout_minus) / (2 * epsilon);
            }

            hessian_.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
            b_ += jacobian_row.transpose() * xout;
            sum += xout.squaredNorm();
        }

        hessian_.template triangularView<Eigen::Upper>() = hessian_.transpose();


        hessian = hessian_.template cast<float>();
        b = b_.template cast<float>();
        return sum;
    }

private:
    inline Eigen::Matrix4d param2Matrix(const VectorN &x) const
    {
        Eigen::Matrix4d transform_matrix_;
        transform_matrix_.setZero();
        transform_matrix_(0, 3) = x[0];
        transform_matrix_(1, 3) = x[1];
        transform_matrix_(2, 3) = x[2];
        transform_matrix_(3, 3) = 1;

        // Compute w from the unit quaternion
        Eigen::Quaternion<double> q(0, x[3], x[4], x[5]);
        q.w() = static_cast<double>(std::sqrt(1 - q.dot(q)));
        q.normalize();
        transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
        return transform_matrix_;
    }
};