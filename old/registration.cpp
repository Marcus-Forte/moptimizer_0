#include <duna/impl/registration.hpp>

#include "duna_exports.h"

using namespace duna;

template class DUNA_OPTIMIZATOR_EXPORT Registration<6,pcl::PointXYZ,pcl::PointXYZ>;
template class DUNA_OPTIMIZATOR_EXPORT Registration<6,pcl::PointXYZ,pcl::PointNormal>;
template class DUNA_OPTIMIZATOR_EXPORT Registration<6,pcl::PointNormal,pcl::PointNormal>;

template class DUNA_OPTIMIZATOR_EXPORT Registration<3,pcl::PointXYZ,pcl::PointXYZ>;
template class DUNA_OPTIMIZATOR_EXPORT Registration<3,pcl::PointXYZ,pcl::PointNormal>;
template class DUNA_OPTIMIZATOR_EXPORT Registration<3,pcl::PointNormal,pcl::PointNormal>;

// For LidAR 
template class DUNA_OPTIMIZATOR_EXPORT Registration<3,pcl::PointXYZI,pcl::PointXYZINormal>;
template class DUNA_OPTIMIZATOR_EXPORT Registration<3,pcl::PointXYZINormal,pcl::PointXYZINormal>;

template class DUNA_OPTIMIZATOR_EXPORT Registration<6,pcl::PointXYZI,pcl::PointXYZINormal>;
template class DUNA_OPTIMIZATOR_EXPORT Registration<6,pcl::PointXYZINormal,pcl::PointXYZINormal>;
