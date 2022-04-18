cmake_minimum_required(VERSION 3.10)

project(duna_optimizator VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INSTALL_PREFIX install) # install locally

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS io common search registration)
find_package(GTest REQUIRED)
link_libraries(Eigen3::Eigen)
include_directories(include src gtest ${PCL_INCLUDE_DIRS})

if(UNIX)
# GTest required pthread in UNIX Os
link_libraries(pthread)
endif()

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set (CMAKE_BUILD_TYPE RELEASE)
#add_compile_options(-msse2 -msse3)

# duna libraries
file (GLOB duna_opt_src "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${duna_opt_src})

include (GenerateExportHeader) 
generate_export_header(${PROJECT_NAME}
EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/exports/duna_exports.h)
include_directories(${CMAKE_BINARY_DIR}/exports)

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

get_filename_component(DATASET_PATH gtest/data/ REALPATH)
add_definitions(-DTEST_DATA_DIR="${DATASET_PATH}")

#gtest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
enable_testing()
include(GoogleTest)


# Make test cases
file(GLOB test_src "gtest/*.cpp")
foreach(test_case ${test_src})
        get_filename_component(test_src_name ${test_case} NAME_WE)
        message(STATUS ${filename})    
        add_executable(${test_src_name} ${test_case})
        target_link_libraries(${test_src_name} ${PCL_LIBRARIES} ${PROJECT_NAME} ${GTEST_LIBRARIES})
        gtest_discover_tests(${test_src_name})
endforeach()

#set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${header_files}")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
)

# TODO how to make sure client code will access header correctly? How to link with library ?
install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
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

