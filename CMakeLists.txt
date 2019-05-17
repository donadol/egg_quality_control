CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
# Enable C++11 for mac
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(OpenCV_DIR /usr/local/Cellar/OpenCV_installation/installation/OpenCV-master/lib/cmake/opencv4)
## ================
## = Project name =
## ================
PROJECT( pr_egg_quality_control )

## ============
## = Packages =
## ============

FIND_PACKAGE( OpenCV REQUIRED )


ADD_EXECUTABLE( pr_egg_quality_control egg_quality_control.cxx )

TARGET_LINK_LIBRARIES( pr_egg_quality_control ${OpenCV_LIBS} )

