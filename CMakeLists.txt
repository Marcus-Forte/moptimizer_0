cmake_minimum_required(VERSION 3.10)
project(duna-optimizer VERSION 0.1 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_EXPERIMENTAL "Build experimental programs" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
link_libraries(Eigen3::Eigen)

if(UNIX)
# GTest required pthread in UNIX Os
        link_libraries(pthread)
        if(LINUX) # not APPLE
        add_compile_options(-msse -msse2 -mavx -mfma -msse4)
        add_compile_options(-march=native)
        endif()
elseif(MSVC)
        add_compile_options(/bigobj)
        add_compile_definitions(_USE_MATH_DEFINES)
endif()

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(${PROJECT_NAME} SHARED
src/levenberg_marquadt.cpp
src/levenberg_marquadt_dynamic.cpp #temporarily removed
src/cost_function_analytical_dynamic.cpp
src/cost_function_numerical_dynamic.cpp
src/covariance.cpp
src/so3.cpp
src/exception.cpp
src/logger.cpp
)

target_link_libraries(${PROJECT_NAME} TBB::tbb)

include (GenerateExportHeader) 
generate_export_header(${PROJECT_NAME}
EXPORT_FILE_NAME ${CMAKE_SOURCE_DIR}/include/duna_optimizer/duna_exports.h)

if(BUILD_EXPERIMENTAL)
        add_subdirectory(experimental EXCLUDE_FROM_ALL)
endif()


#gtest
if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  enable_testing()
  include(GoogleTest)
  # Make test cases
  add_executable(
  optimizer_tests
  tst/test_main.cpp
  tst/simple_model.cpp    
  tst/camera_calibration.cpp  
  tst/manifold.cpp
  tst/powell.cpp
  tst/curve_fitting.cpp
  tst/multiple_objectives.cpp
  tst/loss_function.cpp
  tst/logging.cpp
  tst/covariance.cpp
  tst/differentiation.cpp
  )

  target_link_libraries(optimizer_tests ${GTEST_LIBRARIES} ${PROJECT_NAME} )
  gtest_discover_tests(optimizer_tests)
endif()

# Link library to headers, but does not install. 
target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# --- Installation process --- #
include(GNUInstallDirs)
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install 
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}      
)

# Targets file
install(
        EXPORT ${PROJECT_NAME}-targets
        NAMESPACE duna::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
# Version
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${PROJECT_NAME}Version.cmake
        VERSION 1.0.0
        COMPATIBILITY SameMajorVersion)
# Config
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake
               @ONLY)

install(FILES 
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        )

# Packaging
if(UNIX)
        set(CPACK_GENERATOR DEB)
        set(CPACK_DEBIAN_PACKAGE_NAME "libdunaoptimizer-dev")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "libeigen3-dev, libtbb-dev")
        set(CPACK_PACKAGE_CONTACT "marcus.forte@dunasystem.com")
        include(CPack)
elseif(WIN32)
        set(CPACK_GENERATOR NSIS)
        include(CPack)
endif()


