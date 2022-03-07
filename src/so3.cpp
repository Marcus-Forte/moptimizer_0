#include "duna/so3.h"

namespace so3
{
    template <typename Scalar>
    void param2Matrix(const Eigen::Matrix<Scalar, 6, 1> &x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_)
    {

        transform_matrix_.setZero();
        transform_matrix_(0, 3) = x[0];
        transform_matrix_(1, 3) = x[1];
        transform_matrix_(2, 3) = x[2];
        transform_matrix_(3, 3) = 1;

        // Compute w from the unit quaternion
        Eigen::Quaternion<Scalar> q(0, x[3], x[4], x[5]);
        q.w() = static_cast<Scalar>(std::sqrt(1 - q.dot(q)));
        q.normalize();
        transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
    }

    template <typename Scalar>
    void param2Matrix(const Eigen::Matrix<Scalar, 3, 1> &x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_)
    {

        transform_matrix_.setZero();
        transform_matrix_(3,3) = 1;
        // Compute w from the unit quaternion
        Eigen::Quaternion<Scalar> q(0, x[0], x[1], x[2]);
        q.w() = static_cast<Scalar>(std::sqrt(1 - q.dot(q)));
        q.normalize();
        transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
    }

    template <typename Scalar>
    void matrix2Param(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x){
        x[0] = transform_matrix_(0,3);
        x[1] = transform_matrix_(1,3);
        x[2] = transform_matrix_(2,3);

        Eigen::Matrix<Scalar,3,1> ea;

        Eigen::Matrix<Scalar,3,3> rot = transform_matrix_.topLeftCorner(3,3);
        ea = rot.eulerAngles(2,1,0);

        x[3] = ea[0];
        x[4] = ea[1];//;ea[1];
        x[5] = ea[2];//ea[2];
    }

    // 


    // Instantiation
    
    template void so3::param2Matrix<double>(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix<double, 4, 4> &transform_matrix_);
    template void so3::param2Matrix<float>(const Eigen::Matrix<float, 6, 1> &x, Eigen::Matrix<float, 4, 4> &transform_matrix_);

    // 3DOF
    template void so3::param2Matrix<double>(const Eigen::Matrix<double, 3, 1> &x, Eigen::Matrix<double, 4, 4> &transform_matrix_);
    template void so3::param2Matrix<float>(const Eigen::Matrix<float, 3, 1> &x, Eigen::Matrix<float, 4, 4> &transform_matrix_);

    template void so3::matrix2Param<double>(const Eigen::Matrix<double, 4, 4> &transform_matrix_, Eigen::Matrix<double, 6, 1> &x);
    template void so3::matrix2Param<float>(const Eigen::Matrix<float, 4, 4> &transform_matrix_, Eigen::Matrix<float, 6, 1> &x);

}

