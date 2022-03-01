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
    Eigen::Matrix<float, 3, 4> CameraModel;
    Eigen::Matrix4f camera_lidar_frame;
};

template <int NPARAM>
class CalibrationCost : public CostFunction<NPARAM>
{
public:
    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using VectorX = typename CostFunction<NPARAM>::VectorX;
    using MatrixX = typename CostFunction<NPARAM>::MatrixX;
    using CostFunction<NPARAM>::m_dataset;
    using CostFunction<NPARAM>::m_data_size;

    CalibrationCost(unsigned int data_size, void *dataset) : CostFunction<NPARAM>(data_size, dataset) {}
    virtual ~CalibrationCost() = default;

    // Computes error
    double f(const VectorN &xi, VectorX &xout)
    {
        camera_calibration_data_t *l_dataset = reinterpret_cast<camera_calibration_data_t *>(m_dataset);
        
        // assert(m_dataset->point_list.size() == m_dataset->pixel_list.size());

        if (l_dataset->point_list.size() != l_dataset->pixel_list.size())
        {

            throw std::runtime_error("dataset with different sizes!");
        }

        if (l_dataset->point_list.size() == 0 || l_dataset->pixel_list.size() == 0)
        {
            throw std::runtime_error("empty dataset!");
        }

        // Build matrix from xi
        Eigen::Matrix4f transform = Param2Matrix(xi);
        double sum = 0.0;

        for (int i = 0; i < l_dataset->point_list.size(); ++i)
        {

            Eigen::Vector3f out_pixel;
            // Eigen::Vector4f out_pixel;
            // std::cout << "in_pt: "  << m_dataset->point_list[i].getVector4fMap() << std::endl;
            // Model : pixel = CamModel * camera_lidar_frame * point_i : 3x1 = 3x4 * 4x4 * 4x1
            out_pixel = l_dataset->CameraModel * transform * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap();

            // compose error vector
            xout[2 * i] = l_dataset->pixel_list[i].x - (out_pixel[0] / out_pixel[2]);
            xout[2 * i + 1] = l_dataset->pixel_list[i].y - (out_pixel[1] / out_pixel[2]);
            // std::cout << "out_pixel:" << out_pixel << std::endl;
            sum += xout[2 * i]*xout[2 * i] + xout[2 * i + 1]*xout[2 * i + 1];
        }

        return sum;
    }

    // Computes jacobian
    void df(const VectorN &x, MatrixX &xout)
    {
        const float epsilon = 0.00001;

        VectorX f_(xout.rows());
        f(x, f_);

        VectorX f_plus(xout.rows());
        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            x_plus[j] += epsilon;
            f(x_plus, f_plus);

            xout.col(j) = (f_plus - f_) / epsilon;
        }
    }

private:
    Eigen::Matrix4f Param2Matrix(const VectorN &x)
    {
        Eigen::Matrix4f transform_matrix_;
        transform_matrix_.setZero();
        transform_matrix_(0, 3) = x[0];
        transform_matrix_(1, 3) = x[1];
        transform_matrix_(2, 3) = x[2];
        transform_matrix_(3, 3) = 1;

        // Compute w from the unit quaternion
        Eigen::Quaternion<float> q(0, x[3], x[4], x[5]);
        q.w() = static_cast<float>(std::sqrt(1 - q.dot(q)));
        q.normalize();
        transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
        return transform_matrix_;
    }
};