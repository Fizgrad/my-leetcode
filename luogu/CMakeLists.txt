cmake_minimum_required(VERSION 3.24)
project(luogu)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(luogu
        ../luogu_B3630.cpp ../luogu_P1068.cpp ../luogu_B3747.cpp)