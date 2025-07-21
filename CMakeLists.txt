cmake_minimum_required(VERSION 3.15)
project(DispatcherDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器为clang++
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

# 设置默认构建类型为Debug，如果没有指定的话
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# 根据构建类型设置不同的编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug模式：添加调试信息，禁用优化，启用更多警告
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -O0 -DDEBUG")
    message(STATUS "Debug mode: optimizations disabled, debug info enabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release模式：启用优化，去掉调试信息
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -DNDEBUG")
    message(STATUS "Release mode: optimizations enabled")
else()
    # 默认编译选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
endif()

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