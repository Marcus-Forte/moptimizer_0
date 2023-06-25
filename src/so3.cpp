#include "duna_optimizer/so3.h"

#include "duna_optimizer/duna_exports.h"

namespace so3 {
template <typename Scalar>
inline void convert6DOFParameterToMatrix(const Scalar *x,
                                         Eigen::Matrix<Scalar, 4, 4> &transform_matrix_) {
  transform_matrix_.setIdentity();
  transform_matrix_(0, 3) = x[0];
  transform_matrix_(1, 3) = x[1];
  transform_matrix_(2, 3) = x[2];
  transform_matrix_(3, 3) = 1;
  // EXP
  Eigen::Matrix<Scalar, 3, 1> delta(x[3], x[4], x[5]);
  Eigen::Matrix<Scalar, 3, 3> rot;
  Exp<Scalar>(delta, rot);
  transform_matrix_.topLeftCorner(3, 3) = rot;
}

template <typename Scalar>
inline void convert3DOFParameterToMatrix(const Scalar *x,
                                         Eigen::Matrix<Scalar, 4, 4> &transform_matrix_) {
  transform_matrix_.setIdentity();
  transform_matrix_(3, 3) = 1;

  Eigen::Matrix<Scalar, 3, 1> delta(x[0], x[1], x[2]);
  Eigen::Matrix<Scalar, 3, 3> rot;
  Exp<Scalar>(delta, rot);
  transform_matrix_.topLeftCorner(3, 3) = rot;
}

template <typename Scalar>
inline void convert3DOFParameterToMatrix3(const Scalar *x,
                                          Eigen::Matrix<Scalar, 3, 3> &transform_matrix_) {
  transform_matrix_.setIdentity();
  Eigen::Matrix<Scalar, 3, 1> delta(x[0], x[1], x[2]);
  Eigen::Matrix<Scalar, 3, 3> rot;
  Exp<Scalar>(delta, transform_matrix_);
}

template <typename Scalar>
inline void Exp(const Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>> &delta,
                Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> R) {
  Scalar delta_norm = delta.norm();

  // Rodrigues Tranformation
  if (delta_norm > 10.0 * (std::numeric_limits<Scalar>::epsilon())) {
    Eigen::Matrix<Scalar, 3, 1> r_axis = delta / delta_norm;
    Eigen::Matrix<Scalar, 3, 3> K;
    K << SKEW_SYMMETRIC_FROM(r_axis);
    R.noalias() = Eigen::Matrix<Scalar, 3, 3>::Identity() + std::sin(delta_norm) * K +
                  (1.0 - std::cos(delta_norm)) * K * K;
  } else {
    R.noalias() = Eigen::Matrix<Scalar, 3, 3>::Identity();
  }
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> Exp(const Eigen::Matrix<Scalar, 3, 1> &ang) {
  Scalar ang_norm = ang.norm();
  Eigen::Matrix<Scalar, 3, 3> Eye3 = Eigen::Matrix<Scalar, 3, 3>::Identity();
  if (ang_norm > 0.0000001) {
    Eigen::Matrix<Scalar, 3, 1> r_axis = ang / ang_norm;
    Eigen::Matrix<Scalar, 3, 3> K;
    K << SKEW_SYMMETRIC_FROM(r_axis);

    // Rodrigues Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  } else {
    return Eye3;
  }
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> Exp(const Eigen::Matrix<Scalar, 3, 1> &ang_vel, const Scalar &dt) {
  Scalar ang_vel_norm = ang_vel.norm();
  Eigen::Matrix<Scalar, 3, 3> Eye3 = Eigen::Matrix<Scalar, 3, 3>::Identity();

  if (ang_vel_norm > 0.0000001) {
    Eigen::Matrix<Scalar, 3, 1> r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix<Scalar, 3, 3> K;

    K << SKEW_SYMMETRIC_FROM(r_axis);

    Scalar r_ang = ang_vel_norm * dt;

    // Rodrigues Tranformation
    return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  } else {
    return Eye3;
  }
}

template <typename Scalar>
inline void Log(const Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> &R,
                Eigen::Matrix<Scalar, 3, 1> &delta) {
  Scalar theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Matrix<Scalar, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (std::abs(theta) < 0.001) {
    delta = (0.5 * K);
  } else {
    delta = (0.5 * theta / std::sin(theta) * K);
  }
}

template <typename Scalar>
void inverseRightJacobian(const Eigen::Matrix<Scalar, 3, 1> &r,
                          Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> inv_jacobian) {
  double theta_sq = r.dot(r);
  if (theta_sq < 1e-5) {
    inv_jacobian = Eigen::Matrix<Scalar, 3, 3>::Identity();
    return;
  }

  Eigen::Matrix<Scalar, 3, 3> r_skew;
  r_skew << SKEW_SYMMETRIC_FROM(r);

  Scalar factor = 1 / r.squaredNorm() - (1 + cos(r.norm())) / (2 * r.norm() * sin(r.norm()));
  inv_jacobian = Eigen::Matrix<Scalar, 3, 3>::Identity() + 0.5 * r_skew + factor * r_skew * r_skew;
}

template <typename Scalar>
void rightJacobian(const Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>> &r,
                   Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> right_jacobian) {
  double theta_sq = r.dot(r);

  if (theta_sq < 1e-5) {
    right_jacobian = Eigen::Matrix<Scalar, 3, 3>::Identity();
    return;
  }

  Eigen::Matrix<Scalar, 3, 3> r_skew;
  r_skew << SKEW_SYMMETRIC_FROM(r);

  Scalar factor = (1.0 - cos(r.norm())) / theta_sq;
  right_jacobian = Eigen::Matrix<Scalar, 3, 3>::Identity() - factor * r_skew;
}

template <typename Scalar>
void leftJacobian(const Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>> &r,
                  Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> left_jacobian) {
  double theta_sq = r.dot(r);

  if (theta_sq < 1e-5) {
    left_jacobian = Eigen::Matrix<Scalar, 3, 3>::Identity();
    return;
  }

  Eigen::Matrix<Scalar, 3, 3> r_skew;
  r_skew << SKEW_SYMMETRIC_FROM(r);

  Scalar factor = (1.0 - cos(r.norm())) / theta_sq;
  left_jacobian = Eigen::Matrix<Scalar, 3, 3>::Identity() + factor * r_skew;
}

template void DUNA_OPTIMIZER_EXPORT convert6DOFParameterToMatrix<double>(
    const double *x, Eigen::Matrix<double, 4, 4> &transform_matrix_);
template void DUNA_OPTIMIZER_EXPORT
convert6DOFParameterToMatrix<float>(const float *x, Eigen::Matrix<float, 4, 4> &transform_matrix_);

template void DUNA_OPTIMIZER_EXPORT convert3DOFParameterToMatrix<double>(
    const double *x, Eigen::Matrix<double, 4, 4> &transform_matrix_);
template void DUNA_OPTIMIZER_EXPORT
convert3DOFParameterToMatrix<float>(const float *x, Eigen::Matrix<float, 4, 4> &transform_matrix_);

template void DUNA_OPTIMIZER_EXPORT
convert3DOFParameterToMatrix3<float>(const float *x, Eigen::Matrix<float, 3, 3> &transform_matrix_);
template void DUNA_OPTIMIZER_EXPORT convert3DOFParameterToMatrix3<double>(
    const double *x, Eigen::Matrix<double, 3, 3> &transform_matrix_);

template void DUNA_OPTIMIZER_EXPORT
Exp<double>(const Eigen::Ref<const Eigen::Matrix<double, 3, 1>> &delta,
            Eigen::Ref<Eigen::Matrix<double, 3, 3>> R);
template void DUNA_OPTIMIZER_EXPORT
Exp<float>(const Eigen::Ref<const Eigen::Matrix<float, 3, 1>> &delta,
           Eigen::Ref<Eigen::Matrix<float, 3, 3>> R);

template Eigen::Matrix<float, 3, 3> DUNA_OPTIMIZER_EXPORT
Exp<float>(const Eigen::Matrix<float, 3, 1> &ang);
template Eigen::Matrix<double, 3, 3> DUNA_OPTIMIZER_EXPORT
Exp<double>(const Eigen::Matrix<double, 3, 1> &ang);

template Eigen::Matrix<float, 3, 3> DUNA_OPTIMIZER_EXPORT
Exp<float>(const Eigen::Matrix<float, 3, 1> &ang_vel, const float &dt);
template Eigen::Matrix<double, 3, 3> DUNA_OPTIMIZER_EXPORT
Exp<double>(const Eigen::Matrix<double, 3, 1> &ang_vel, const double &dt);

template void DUNA_OPTIMIZER_EXPORT Log<double>(const Eigen::Ref<Eigen::Matrix<double, 3, 3>> &R,
                                                Eigen::Matrix<double, 3, 1> &delta);
template void DUNA_OPTIMIZER_EXPORT Log<float>(const Eigen::Ref<Eigen::Matrix<float, 3, 3>> &R,
                                               Eigen::Matrix<float, 3, 1> &delta);

template void DUNA_OPTIMIZER_EXPORT inverseRightJacobian<float>(
    const Eigen::Matrix<float, 3, 1> &r, Eigen::Ref<Eigen::Matrix<float, 3, 3>> inv_jacobian);
template void DUNA_OPTIMIZER_EXPORT inverseRightJacobian<double>(
    const Eigen::Matrix<double, 3, 1> &r, Eigen::Ref<Eigen::Matrix<double, 3, 3>> inv_jacobian);

template void DUNA_OPTIMIZER_EXPORT
rightJacobian<float>(const Eigen::Ref<const Eigen::Matrix<float, 3, 1>> &r,
                     Eigen::Ref<Eigen::Matrix<float, 3, 3>> jacobian);
template void DUNA_OPTIMIZER_EXPORT
rightJacobian<double>(const Eigen::Ref<const Eigen::Matrix<double, 3, 1>> &r,
                      Eigen::Ref<Eigen::Matrix<double, 3, 3>> jacobian);

template void DUNA_OPTIMIZER_EXPORT
leftJacobian<float>(const Eigen::Ref<const Eigen::Matrix<float, 3, 1>> &r,
                    Eigen::Ref<Eigen::Matrix<float, 3, 3>> jacobian);
template void DUNA_OPTIMIZER_EXPORT
leftJacobian<double>(const Eigen::Ref<const Eigen::Matrix<double, 3, 1>> &r,
                     Eigen::Ref<Eigen::Matrix<double, 3, 3>> jacobian);
}  // namespace so3
