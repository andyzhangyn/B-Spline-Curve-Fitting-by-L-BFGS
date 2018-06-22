cmake_minimum_required(VERSION 3.10)
project(DeepMotion C)

set(CMAKE_C_STANDARD 11)

add_executable(DeepMotion
        SplineDrawer-master/spline_to_bezier.m)
