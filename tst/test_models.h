#pragma once

#include "moptimizer/model.h"

// Function to be minimized

template <class Scalar>
struct Model : public moptimizer::BaseModel<Scalar, Model<Scalar>> {
  Model(Scalar *x, Scalar *y) : data_x(x), data_y(y) {}
  // API simply has to override this method

  bool f(const Scalar *x, Scalar *residual, unsigned int index) const override {
    residual[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
    return true;
  }

 private:
  const Scalar *const data_x;
  const Scalar *const data_y;
};