cmake_minimum_required (VERSION 3.11.3)

# set(CMAKE_CXX_STANDARD 17)
project(ObjectClassification) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -pthread")

#use compiled version
SET(OpenCV_DIR /usr/local/Cellar/OpenCV_installation/installation/OpenCV-master/lib/cmake/opencv4)

# Requires OpenCV 
find_package(OpenCV REQUIRED)

# show a message with opencv version detected
message("OpenCV version : ${OpenCV_VERSION}")

#dd to our environment the headers and the directory of the specified library
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h

# create executable
add_executable(${PROJECT_NAME} ${project_SRCS})
 
 #link our libary
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})






