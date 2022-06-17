cmake_minimum_required(VERSION 3.10)
project(duna_optimizer VERSION 0.1 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_EXPERIMENTAL "Build experimental programs" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set(CMAKE_INSTALL_PREFIX install) # always install locally

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS io common search registration)
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
link_libraries(Eigen3::Eigen OpenMP::OpenMP_CXX)
include_directories(include gtest ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})



# add_compile_options(-march=native)
# add_compile_options(-msse -msse2 -mavx -mfma -msse4)

if(UNIX)
# GTest required pthread in UNIX Os
        link_libraries(pthread)
endif()

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(${PROJECT_NAME} SHARED
src/levenberg_marquadt.cpp
src/cost_function.cpp
src/so3.cpp
src/transformation_estimation6DOF.cpp
src/transformation_estimation3DOF.cpp
)


include (GenerateExportHeader) 
generate_export_header(${PROJECT_NAME}
EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/exports/duna_exports.h)
include_directories(${CMAKE_BINARY_DIR}/exports)

# target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
if(BUILD_EXPERIMENTAL)
        add_subdirectory(experimental EXCLUDE_FROM_ALL)
endif()


#gtest
if(BUILD_TESTS)
        # Copy point cloud
        get_filename_component(DATASET_PATH test/data/ REALPATH)
        add_definitions(-DTEST_DATA_DIR="${DATASET_PATH}")
        
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        enable_testing()
        include(GoogleTest)

        # Make test cases
        set(test_src 
        test/simple_model.cpp    
        test/camera_calibration.cpp  
        test/differentiation.cpp
        test/manifold.cpp
        test/powell.cpp
        test/curve_fitting.cpp

        # REG
        test/registration_point2plane.cpp
        test/registration_point2point.cpp
        test/registration_point2plane3DOF.cpp

        # Sequence
        test/sequence_3dof.cpp
        )

        foreach(test_case ${test_src})
                get_filename_component(test_src_name ${test_case} NAME_WE)
                message(STATUS ${filename})    
                add_executable(${test_src_name} ${test_case})
                target_link_libraries(${test_src_name} ${PCL_LIBRARIES} ${PROJECT_NAME} ${GTEST_LIBRARIES})
                gtest_discover_tests(${test_src_name})
        endforeach()

        add_executable(draft test/draft.cpp)
        target_link_libraries(draft ${GTEST_LIBRARIES})
endif()

# Does this change anything ?
target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

# Packaging
if(UNIX)
        set(CPACK_GENERATOR DEB)
        set(CPACK_DEBIAN_PACKAGE_NAME "libdunaoptimizator-dev")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "libpcl-dev,libeigen3-dev")
        set(CPACK_PACKAGE_CONTACT "marcus.forte@dunasystem.com")
        include(CPack)
elseif(WIN32)
        set(CPACK_GENERATOR NSIS)
        include(CPack)
endif()

