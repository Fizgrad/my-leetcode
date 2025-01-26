cmake_minimum_required(VERSION 3.10)
project(LeetCodeSolutions)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")

# 包含 src 目录
include_directories(src)

# 设置 CMake 使用 Clang 编译器
set(CMAKE_CXX_COMPILER "clang++")

# 自动为 src 目录中的每个 cpp 文件创建一个可执行目标
file(GLOB CPP_SOURCES src/*.cpp)
foreach(CPP_SOURCE ${CPP_SOURCES})
    # 获取不带路径的文件名
    get_filename_component(SOURCE_NAME ${CPP_SOURCE} NAME_WE)
    # 添加可执行文件
    add_executable(${SOURCE_NAME} ${CPP_SOURCE})
endforeach()
