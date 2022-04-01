#pragma once

#include "duna/cost_function.h"
#include "duna/so3.h"

#include <Eigen/Dense>
#include <vector>
#include <pcl/point_types.h>

namespace duna
{

    /* Define your dataset structure*/
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
        using VectorNd = Eigen::Matrix<double, NPARAM, 1>;

        using CostFunction<NPARAM>::m_dataset;
        CalibrationCost(void *dataset) : CostFunction<NPARAM>(dataset)
        {
            l_dataset = reinterpret_cast<camera_calibration_data_t *>(m_dataset);
        }
        virtual ~CalibrationCost() = default;

        virtual void checkData() override
        {
            if (l_dataset->point_list.size() == 0)
                throw std::runtime_error("Empty point list.");

            if (l_dataset->pixel_list.size() == 0)
                throw std::runtime_error("Empty pixel list.");

            if (l_dataset->pixel_list.size() != l_dataset->point_list.size())
            {
                std::stringstream msg;
                msg << "Pixel list and point list with different sizes. (" << std::to_string(l_dataset->pixel_list.size()) << ") != "
                    << "(" << l_dataset->point_list.size() << ")";

                throw std::runtime_error(msg.str());
            }
        }

        double computeCost(const VectorN &x) override
        {

            double sum = 0;

            Eigen::Matrix4d transform;
            VectorNd x_double(x.template cast<double>());

            // Eigen::Matrix4f transform;

            so3::param2Matrix(x_double, transform);

            for (int i = 0; i < l_dataset->point_list.size(); ++i)
            {

                Eigen::Vector3d out_pixel;
                out_pixel = l_dataset->CameraModel * transform * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast<double>();

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

            double sum = 0;
            hessian.setZero();
            b.setZero();

            // Build matrix from xi
            Eigen::Matrix4d transform;

            VectorNd x_double;
            for (int i = 0; i < NPARAM; ++i)
            {
                x_double[i] = x[i];
            }

            so3::param2Matrix<double>(x_double, transform);

            Eigen::Matrix<double, 2, NPARAM> jacobian_row;

            // Build incremental transformations
            Eigen::Matrix4d transform_plus[NPARAM];
            Eigen::Matrix4d transform_minus[NPARAM];

            Eigen::Matrix<double, NPARAM, NPARAM> hessian_;
            Eigen::Matrix<double, NPARAM, 1> b_;
            hessian_.setZero();
            b_.setZero();

            const double epsilon = 1e-6;
            for (int j = 0; j < NPARAM; ++j)
            {
                VectorNd x_plus(x_double);
                VectorNd x_minus(x_double);
                x_plus[j] += epsilon;
                x_minus[j] -= epsilon;

                so3::param2Matrix<double>(x_plus, transform_plus[j]);
                so3::param2Matrix<double>(x_minus, transform_minus[j]);
            }

            for (int i = 0; i < l_dataset->point_list.size(); ++i)
            {

                Eigen::Vector3d out_pixel;
                out_pixel = l_dataset->CameraModel * transform * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast<double>();

                // compose error vector
                Eigen::Vector2d xout;
                xout[0] = l_dataset->pixel_list[i].x - (out_pixel[0] / out_pixel[2]);
                xout[1] = l_dataset->pixel_list[i].y - (out_pixel[1] / out_pixel[2]);

                for (int j = 0; j < NPARAM; ++j)
                {

                    Eigen::Vector3d out_pixel_plus, out_pixel_minus;
                    out_pixel_plus = l_dataset->CameraModel * transform_plus[j] * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast<double>();
                    out_pixel_minus = l_dataset->CameraModel * transform_minus[j] * l_dataset->camera_lidar_frame * l_dataset->point_list[i].getVector4fMap().cast<double>();

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
        camera_calibration_data_t *l_dataset;
    };

}