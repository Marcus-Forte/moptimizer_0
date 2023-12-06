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
        if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT APPLE)
        add_compile_options(-msse -msse2 -mavx2 -mfma -msse4.2 -mfpmath=sse)
        add_compile_options(-march=native)
        endif()
elseif(MSVC)
        add_compile_options(/bigobj)
        add_compile_definitions(_USE_MATH_DEFINES)
endif()

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(${PROJECT_NAME} SHARED
src/levenberg_marquadt_dyn.cpp 
src/cost_function_analytical_dyn.cpp
src/cost_function_numerical_dyn.cpp
src/delta.cpp
src/so3.cpp
src/exception.cpp
src/logger.cpp
src/linearization.cpp
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

  file (GLOB tests_srcs ${CMAKE_SOURCE_DIR}/tst/*.cpp)
  list (FILTER tests_srcs EXCLUDE REGEX ".draft.cpp")
  # Make test cases
  add_executable(
  optimizer_tests
  ${tests_srcs}
  )

  set(test_data_path ${CMAKE_SOURCE_DIR}/tst/data)
  # cmake_path(GET some_path ROOT_PATH test_data_path)
  message("TEST PATH: ${test_data_path}")
  target_compile_definitions(optimizer_tests PRIVATE -DTEST_DATA_PATH="${test_data_path}" )

  add_executable(draft
  ${CMAKE_SOURCE_DIR}/tst/draft.cpp)
  target_link_libraries(draft ${PROJECT_NAME})
  target_link_libraries(optimizer_tests ${PROJECT_NAME} GTest::gtest GTest::gmock)
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

# set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
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