#include <moptimizer/manifold.h>
#include <moptimizer/so3.h>
#include <gtest/gtest.h>

TEST(Manifold, EuclideanSpace) {
  EuclideanManifold<3> ManifoldA;
  Manifold<3>::TangentRepresentation delta;

  ManifoldA.getEunclideanRepresentation().setRandom();

  Manifold<3>::LinearRepresentation originalA(ManifoldA.getEunclideanRepresentation());

  delta.setRandom();

  ManifoldA.Plus(delta);

  for (int i = 0; i < ManifoldA.getEunclideanRepresentation().size(); ++i) {
    EXPECT_EQ(ManifoldA.getEunclideanRepresentation()[i], originalA[i] + delta[i]);
  }

  // reset
  ManifoldA.getEunclideanRepresentation() = originalA;

  ManifoldA.Minus(delta);

  for (int i = 0; i < ManifoldA.getEunclideanRepresentation().size(); ++i) {
    EXPECT_EQ(ManifoldA.getEunclideanRepresentation()[i], originalA[i] - delta[i]);
  }
}

class ManifoldSO3 : public Manifold<3, 9> {
 public:
  using typename Manifold<3, 9>::LinearRepresentation;
  using typename Manifold<3, 9>::TangentRepresentation;
  using Manifold<3, 9>::parameter;

 public:
  ManifoldSO3() : parameter_matrix(parameter.data()) { parameter_matrix.setIdentity(); }

  // Exp3
  void Plus(const TangentRepresentation &rhs) {
    Eigen::Matrix3d &&sum(parameter_matrix);
    so3::Exp<double>(rhs, sum);
    parameter_matrix = parameter_matrix * sum;
  }

  // Log
  void Minus(const TangentRepresentation &rhs) {}

  // Overload parent
  Eigen::Map<Eigen::Matrix3d> &getEunclideanRepresentation() { return parameter_matrix; }

 private:
  Eigen::Map<Eigen::Matrix3d> parameter_matrix;
};

TEST(Manifold, SO3) {
  ManifoldSO3 ManifoldA;
  ManifoldSO3::TangentRepresentation delta;
  delta.setZero();

  delta[0] = 0.02;
  // delta[1] = 0.02;
  // delta[2] = 0.02;

  // ManifoldA.Plus(delta);
  ManifoldA.Plus(delta);

  std::cout << ManifoldA.getEunclideanRepresentation() << std::endl;

  // Non elegant way
  Eigen::Matrix3d NonManifold = Eigen::Matrix3d::Identity();
  Eigen::Quaterniond q(0, delta[0], delta[1], delta[2]);

  q.w() = static_cast<double>(std::sqrt(1 - q.dot(q)));
  q.normalize();

  NonManifold = q.toRotationMatrix();

  std::cout << NonManifold << std::endl;
}
