#!/bin/sh

# 进入上一级目录
cd ..

# 创建 build 文件夹
mkdir -p build

# 进入 build 文件夹
cd build

# 运行 cmake 命令并生成 compile_commands.json 文件
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ../leetcode-cpp

# 检查 cmake 命令是否成功
if [ $? -eq 0 ]; then
    echo "CMake configuration successful. compile_commands.json has been generated."
    make
else
    echo "CMake configuration failed."
fi
