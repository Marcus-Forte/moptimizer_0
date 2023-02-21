#include <gtest/gtest.h>
#include <duna/cost_function_numerical.h>
#include <duna/levenberg_marquadt.h>
#include <duna/levenberg_marquadt_dynamic.h>
#include <duna/model.h>
#include <duna/so3.h>

#define MODEL_PARAMETERS 6
#define MODEL_OUTPUTS 2
#define TOLERANCE 5e-5

struct Model : public duna::BaseModel<double>
{
    Model(const std::vector<Eigen::Vector4d> &point_list, const std::vector<Eigen::Vector2i> &pixel_list)
        : point_vector(point_list), pixel_vector(pixel_list)
    {
        if (pixel_list.empty())
            throw std::runtime_error("Empty pixel list");

        if (point_list.empty())
            throw std::runtime_error("Empty point list");

        if (pixel_list.size() != point_list.size())
            throw std::runtime_error("Different point sizes");

        camera_laser_frame_conversion.setIdentity();

        Eigen::Matrix3d rot;
        rot = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ());
        camera_laser_frame_conversion.block<3, 3>(0, 0) = rot;

        camera_model << 586.122314453125, 0, 638.8477694496105, 0, 0,
            722.3973388671875, 323.031267074588, 0,
            0, 0, 1, 0;
    }

    inline void setup(const double *x) override
    {
        so3::convert6DOFParameterToMatrix(x, transform);
    }

    inline bool f(const double *x, double *residual, unsigned int index) override
    {
        Eigen::Vector3d out_pixel;
        out_pixel = camera_model * transform * camera_laser_frame_conversion * point_vector[index];
        residual[0] = pixel_vector[index][0] - (out_pixel[0] / out_pixel[2]);
        residual[1] = pixel_vector[index][1] - (out_pixel[1] / out_pixel[2]);
        return true;
    }

private:
    // input data

    const std::vector<Eigen::Vector4d> &point_vector;
    const std::vector<Eigen::Vector2i> &pixel_vector;

    Eigen::Matrix<double, 3, 4> camera_model;
    Eigen::Matrix4d camera_laser_frame_conversion;

    // Parameter to matrix conversion
    Eigen::Matrix4d transform;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);

    return RUN_ALL_TESTS();
}

class CameraCalibration : public testing::Test
{
public:
    CameraCalibration()
    {
        // TODO I spent quite some time debugging why vectors would have their value changed after running tests...
        // The vectors were allocated in stack and lost their life after constructor. BAD!

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

        cost = new duna::CostFunctionNumerical<double, 6, 2>(
            Model::Ptr(new Model(point_list, pixel_list)),
            5);
        optimizer.addCost(cost);
    }

    ~CameraCalibration()
    {
        delete cost;
    }

protected:
    duna::CostFunctionNumerical<double, MODEL_PARAMETERS, MODEL_OUTPUTS> *cost;
    duna::LevenbergMarquadt<double, MODEL_PARAMETERS> optimizer;

    std::vector<Eigen::Vector4d> point_list;
    std::vector<Eigen::Vector2i> pixel_list;

    const double matlab_solution[MODEL_PARAMETERS] = {
        -0.010075911761110,
        0.020714594988011,
        -0.058274626693636,
        0.009185665242934,
        -0.000659168270130,
        0.013700587828461,
    };

    const double ceres_solution[MODEL_PARAMETERS] =
        {
            -0.0101064,
            0.0206767,
            -0.0582803,
            0.0183564,
            -0.00130745,
            0.027414};
};

TEST_F(CameraCalibration, GoodWeather)
{
    double x0[6] = {0};

    optimizer.minimize(x0);

    for (int i = 0; i < MODEL_PARAMETERS; ++i)
    {
        EXPECT_NEAR(x0[i], ceres_solution[i], TOLERANCE);
    }

    std::cerr << Eigen::Map<Eigen::Matrix<double, 6, 1>>(x0);
}

TEST_F(CameraCalibration, BadWeather)
{
    double x0[6] = {0.5, 0.5, 0.5, 0.2, 0.5, 0.5};
    optimizer.setMaximumIterations(50);
    optimizer.minimize(x0);

    for (int i = 0; i < MODEL_PARAMETERS; ++i)
    {
        EXPECT_NEAR(x0[i], ceres_solution[i], TOLERANCE);
    }

    std::cerr << Eigen::Map<Eigen::Matrix<double, 6, 1>>(x0);
}

TEST_F(CameraCalibration, GoodWeatherDynamic)
{
    double x0[6] = {0};

    duna::LevenbergMarquadtDynamic<double> dyn_optimizer(6);
    dyn_optimizer.addCost(this->cost);
    dyn_optimizer.minimize(x0);

    for (int i = 0; i < MODEL_PARAMETERS; ++i)
    {
        EXPECT_NEAR(x0[i], ceres_solution[i], TOLERANCE);
    }

    std::cerr << Eigen::Map<Eigen::Matrix<double, 6, 1>>(x0);
}

TEST_F(CameraCalibration, BadWeatherDynamic)
{
    double x0[6] = {0.5, 0.5, 0.5, 0.2, 0.5, 0.5};
    duna::LevenbergMarquadtDynamic<double> dyn_optimizer(6);
    dyn_optimizer.addCost(this->cost);
    dyn_optimizer.setMaximumIterations(50);
    dyn_optimizer.minimize(x0);

    for (int i = 0; i < MODEL_PARAMETERS; ++i)
    {
        EXPECT_NEAR(x0[i], ceres_solution[i], TOLERANCE);
    }

    std::cerr << Eigen::Map<Eigen::Matrix<double, 6, 1>>(x0);
}