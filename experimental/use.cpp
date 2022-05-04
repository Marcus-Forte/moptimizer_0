#include <levenberg_marquadt.h>
#include <cost_function.h>
#include <vector>
#include <iostream>
#include <memory>
using namespace duna;

// model: y(x) = b0*x / (b1 + x)
// i 	1 	2 	3 	4 	5 	6 	7
// [S] 	0.038 	0.194 	0.425 	0.626 	1.253 	2.500 	3.740
// Rate 	0.050 	0.127 	0.094 	0.2122 	0.2729 	0.2665 	0.3317

double x_data[] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
double y_data[] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};


struct model
{
    model(double* x_, double *y_) : data_x(x_), data_y(y_)
    {}
    void operator()(const double*x, double *residual, unsigned int index)
    {
        residual[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    }

    private:
    const double * const data_x;
    const double * const data_y;

};


#define PARAMETERS 2

int main()
{


    CostFunction<model,double,2,1> * cost (new CostFunction<model,double,2,1>(
        new model(x_data,y_data),
        7,1));

    Eigen::Vector2d x0;
    x0[0] = 0.9;
    x0[1] = 0.2;
    
    double sum = cost->computeCost(x0.data());
    Eigen::Matrix2d hessian;
    Eigen::Vector2d residuals;
    double sum_lin = cost->linearize(x0,hessian,residuals);



    std::cout <<sum << std::endl;
    std::cout <<sum_lin << std::endl;
    std::cout <<hessian << std::endl;

    LevenbergMarquadt<double,2,1>  optimizer; // = new LevenbergMarquadt<double>;
    optimizer.setMaximumIterations(15);
    optimizer.setCost(cost);

    optimizer.minimize(x0);

    std::cout << x0 << std::endl;
}