#!/bin/bash

# 构建PyTorch风格Dispatcher演示程序

echo "正在构建 PyTorch风格Dispatcher 演示程序..."

# 创建构建目录
mkdir -p build
cd build

# 配置CMake，使用ninja生成器和clang编译器
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang

# 检查CMake是否成功
if [ $? -ne 0 ]; then
    echo "CMake配置失败"
    exit 1
fi

echo "正在使用ninja编译..."

# 使用ninja编译
ninja

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败"
    exit 1
fi

echo "编译成功！"
echo "可执行文件位置: ./bin/dispatcher_demo"
echo ""
echo "运行演示程序:"
echo "./bin/dispatcher_demo" 