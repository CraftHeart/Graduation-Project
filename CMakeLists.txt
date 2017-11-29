cmake_minimum_required(VERSION 3.8)
project(camera_match)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp src/imgfeatures.c src/kdtree.c
        src/minpq.c src/siftmatch.cpp src/sift.c src/utils.c src/xform.c)

include_directories(
        /home/wsn/Downloads/opencv-2.4.13.4/include
        /home/wsn/Downloads/opencv-2.4.13.4/include/opencv
        /home/wsn/Downloads/opencv-2.4.13.4/include/opencv2
        inc/
)

add_executable(camera_match ${SOURCE_FILES})

target_link_libraries(camera_match -I/usr/local/include/opencv -I/usr/local/include
        -L/usr/local/lib -lopencv_calib3d -lopencv_contrib -lopencv_core
        -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui
        -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree
        -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching
        -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
        -lrt -lpthread -lm -ldl
        )