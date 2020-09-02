cmake_minimum_required(VERSION 2.8)
project(point_syth)

set(CMAKE_CXX_FLAGS "--std=c++11")
# 需要先找到对应的包 然后才能将路径添加进去
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDES_DIRS})

include_directories("/usr/include/eigen3")

# PCL
find_package( PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# g2o
#find_package(G2O REQUIRED)
include_directories("/home/spyder/g2o")

# Sophus
#find_package(Sophus REQUIRED)
include_directories("/home/spyder/Sophus")

# boost
include_directories("/home/spyder/boost_1_73_0")

# add_definitions(${PCL_DEFINITIONS})

# find_package(Pangolin REQUIRED)
add_executable(point-syth point_syth.cpp)

# target_link_libraries(stereo ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(point-syth ${OpenCV_LIBS} ${PCL_LIBRARIES} )
