#!/bin/bash
set -e
g++ compute.cpp src/glad.c -Iinclude -I/usr/include/opencv4 -lglfw3 -lpulse -lpulse-simple -lfftw3f -pthread -ldl -lopencv_core -lopencv_imgproc -lopencv_videoio -o compute "$@"
echo "Build complete: ./compute"
