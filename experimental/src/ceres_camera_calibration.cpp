#include <ceres/ceres.h>
#include <iostream>
#include <duna/so3.h>
using namespace ceres;

template <typename Scalar>
inline void convert6DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_)
{
    transform_matrix_.setZero();
    transform_matrix_(0, 3) = x[0];
    transform_matrix_(1, 3) = x[1];
    transform_matrix_(2, 3) = x[2];
    transform_matrix_(3, 3) = 1;

    // Compute w from the unit quaternion
    Eigen::Quaternion<Scalar> q(0, x[3], x[4], x[5]);

    Scalar &&q_dot_q = q.dot(q);

    // if (q_dot_q > 1)
    // {
    //     q = q.normalized().coeffs() * 0.1;
    //     q_dot_q = q.dot(q);
    //     // std::cerr << "adjusting...\n";
    // }

    q.w() = static_cast<Scalar>(std::sqrt(1 - q_dot_q));
    q.normalize();
    transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
}

struct Model
{

    Model(const Eigen::Matrix<double, 4, 4> &camera_laser_conversion_,
          const Eigen::Matrix<double, 3, 4> &camera_model_,
          const Eigen::Matrix<double, 4, 1> &point_,
          const Eigen::Matrix<int, 2, 1> &pixel_) : camera_laser_conversion(camera_laser_conversion_),
                                                    camera_model(camera_model_),
                                                    point(point_),
                                                    pixel(pixel_)
    {
    }

    template <typename T>
    bool operator()(const T *const x, T *residual) const
    {

        Eigen::Vector3d out_pixel;
        Eigen::Matrix<T, 4, 4> transform;
        convert6DOFParameterToMatrix(x, transform);

        out_pixel = camera_model * transform * camera_laser_conversion * point;

        residual[0] = pixel[0] - (out_pixel[0] / out_pixel[2]);
        residual[1] = pixel[1] - (out_pixel[1] / out_pixel[2]);
        return true;
    }

private:
    const Eigen::Matrix<double, 4, 4> &camera_laser_conversion;
    const Eigen::Matrix<double, 3, 4> &camera_model;
    const Eigen::Matrix<double, 4, 1> &point;
    const Eigen::Matrix<int, 2, 1> &pixel;
};

/// MUST USE MANIFOLDS

int main()

{
    // ProductManifold<EuclideanManifold<3>, QuaternionManifold> manifold;

    Eigen::Matrix4d camera_laser_frame_conversion;
    camera_laser_frame_conversion.setIdentity();
    Eigen::Matrix<double, 3, 4> camera_model;

    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ());
    camera_laser_frame_conversion.block<3, 3>(0, 0) = rot;

    camera_model << 586.122314453125, 0, 638.8477694496105, 0, 0,
        722.3973388671875, 323.031267074588, 0,
        0, 0, 1, 0;

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Problem problem;

    std::vector<Eigen::Vector4d> point_list;
    std::vector<Eigen::Vector2i> pixel_list;
    point_list.push_back(Eigen::Vector4d(2.055643, 0.065643, 0.684357, 1));
    point_list.push_back(Eigen::Vector4d(1.963083, -0.765833, 0.653833, 1));
    point_list.push_back(Eigen::Vector4d(2.927500, 0.707000, 0.125250, 1));
    point_list.push_back(Eigen::Vector4d(2.957833, 0.384667, 0.123667, 1));
    point_list.push_back(Eigen::Vector4d(2.756000, 0.712000, -0.298000, 1));

    pixel_list.push_back(Eigen::Vector2i(621, 67));
    pixel_list.push_back(Eigen::Vector2i(878, 76));
    pixel_list.push_back(Eigen::Vector2i(491, 279));
    pixel_list.push_back(Eigen::Vector2i(559, 282));
    pixel_list.push_back(Eigen::Vector2i(481, 388));

    double x[6] = {0};

    // Model AA(camera_laser_frame_conversion,camera_model,point_list[0],pixel_list[0]);

    for (int i = 0; i < 5; ++i)
    {
        CostFunction *cost_function = new NumericDiffCostFunction<Model, ceres::CENTRAL, 2, 6>(new Model(camera_laser_frame_conversion, camera_model, point_list[i], pixel_list[i]));
        problem.AddResidualBlock(cost_function, nullptr,x);
    }

    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << Eigen::Map<const Eigen::Matrix<double,6,1>>(x) << std::endl;
    // for (int i = 0; i <)
}