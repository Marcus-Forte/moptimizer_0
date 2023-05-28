#include <duna/covariance/covariance.h>

namespace duna::covariance {
/* No Covariance ~ Identity*/
template <class Scalar>
IdentityCovariance<Scalar>::IdentityCovariance(unsigned int dimension)
    : covariance_matrix_(dimension, dimension) {
  covariance_matrix_.setIdentity();
}

template <class Scalar>
IdentityCovariance<Scalar>::~IdentityCovariance() = default;

template <class Scalar>
inline typename IdentityCovariance<Scalar>::MatrixType
IdentityCovariance<Scalar>::getCovariance(Scalar *input) {
  return covariance_matrix_;
}

// Instantiations
template class IdentityCovariance<float>;
template class IdentityCovariance<double>;
}  // namespace duna::covariance