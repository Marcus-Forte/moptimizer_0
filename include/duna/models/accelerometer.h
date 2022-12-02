#pragma once
#include <duna/model.h>
#include <duna/so3.h>
#include <Eigen/Dense>

namespace duna
{

    class Accelerometer : public BaseModelJacobian<double>
    {
    public:
        Accelerometer() : gravity_(0, 0, 1)
        {
            transform_.setIdentity();
        }

        void setup(const double *x) override
        {
            Eigen::Map<const Eigen::Vector3d> x_map(x);

            so3::Exp<double>(x_map, transform_);
        }

        virtual bool f(const double *x, double *f_x, unsigned int index) override
        {
            // Rotate gravity
            Eigen::Vector3d rotated_gravity = transform_ * gravity_;

            // f_x should be the difference between what was measured by ACC and estimated state!
            f_x[0] = rotated_gravity[0];
            f_x[1] = rotated_gravity[1];
            f_x[2] = rotated_gravity[2];

            return true;
        }

        bool f_df(const double *x, double *f_x, double *jacobian, unsigned int index) override
        {
            f(x,f_x, index);

            // Fill jacobian (3x3)
            return true;
        }

    private:
        Eigen::Matrix3d transform_;
        Eigen::Vector3d gravity_;
    };
} // namespace