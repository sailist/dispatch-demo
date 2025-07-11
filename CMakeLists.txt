cmake_minimum_required(VERSION 3.15)
project(DispatcherDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器为clang++
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 头文件
set(HEADERS
    include/DispatchKey.h
    include/DispatchKeySet.h
    include/IValue.h
    include/TensorImpl.h
    include/OperatorHandle.h
    include/Dispatcher.h
)

# 源文件
set(SOURCES
    src/DispatchKeySet.cpp
    src/IValue.cpp
    src/TensorImpl.cpp
    src/OperatorHandle.cpp
    src/Dispatcher.cpp
    src/main.cpp
)

# 创建可执行文件
add_executable(dispatcher_demo ${SOURCES} ${HEADERS})

# 设置输出目录
set_target_properties(dispatcher_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
) 