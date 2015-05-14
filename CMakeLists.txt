cmake_minimum_required(VERSION 3.1)

project (canvas)

include(CheckIncludeFiles)

check_include_files(CL/cl.h _)
check_include_files(OpenImageIO/imageio.h _)

add_compile_options(-Wall -std=c++11)

add_executable(main main.cc)
target_link_libraries(main OpenCL OpenImageIO)
