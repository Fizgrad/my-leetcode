#!/bin/sh

cd ..

mkdir -p build

cd build

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -G Ninja ../leetcode-c

if [ $? -eq 0 ]; then
    echo "CMake configuration successful. compile_commands.json has been generated."
    ninja
else
    echo "CMake configuration failed."
fi