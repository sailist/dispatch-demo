# PyTorch风格Dispatcher C++实现

这是一个基于 [PyTorch Dispatcher论文](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) 学习PyTorch dispatch机制的demo，演示了现代深度学习框架中分发机制的核心概念。

## 项目特点

- 🚀 **多重分发**: 支持基于多个dispatch key的函数分发
- 📦 **Boxing/Unboxing**: 统一的函数调用约定支持
- 🔧 **Backend抽象**: CPU、CUDA等后端的透明切换  
- 🧩 **功能性包装**: Autograd、Tracing、Profiling等交叉关注点
- 📊 **性能监控**: 内置的操作符调用统计
- 🧵 **线程安全**: 支持多线程环境下的并发访问

## 核心概念

### Dispatch Key
```cpp
enum class DispatchKey {
    CPU,        // CPU后端实现
    CUDA,       // CUDA后端实现  
    Autograd,   // 自动微分包装器
    Tracing,    // JIT追踪包装器
    Profiling,  // 性能监控包装器
    CatchAll    // 兜底实现
};
```

### Dispatch Key Set
使用bitset高效管理dispatch key集合，支持优先级排序：
- 功能性keys (Autograd, Tracing, Profiling) 具有更高优先级
- Backend keys (CPU, CUDA) 提供具体实现

### Boxing机制
```cpp
// 统一的IValue类型支持任意参数类型
IValue cpu_tensor = IValue(make_tensor_cpu({2, 3}));
IValue result = callOp("add", {cpu_tensor, cpu_tensor});
```

## 项目结构

```
dispatcher-demo/
├── CMakeLists.txt          # CMake构建配置
├── build.sh               # 构建脚本
├── README.md              # 项目文档
├── include/               # 头文件目录
│   ├── DispatchKey.h      # Dispatch key枚举定义
│   ├── DispatchKeySet.h   # Dispatch key集合管理
│   ├── IValue.h           # Boxing/Unboxing机制
│   ├── TensorImpl.h       # 简化的Tensor实现
│   ├── OperatorHandle.h   # 操作符句柄和函数表
│   └── Dispatcher.h       # 核心分发器
└── src/                   # 源文件目录
    ├── DispatchKeySet.cpp # Key集合实现
    ├── IValue.cpp         # IValue类实现
    ├── TensorImpl.cpp     # Tensor实现
    ├── OperatorHandle.cpp # 操作符句柄实现
    ├── Dispatcher.cpp     # 分发器核心逻辑
    └── main.cpp           # 演示程序
```

## 构建说明

### 依赖要求
- CMake 3.15+
- Clang编译器 (支持C++17)
- Ninja构建系统

### 编译步骤
```bash
# 给构建脚本执行权限
chmod +x build.sh

# 运行构建脚本
./build.sh

# 运行演示程序
cd build && ./bin/dispatcher_demo
```

### 手动构建
```bash
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_CXX_COMPILER=clang++
ninja
./bin/dispatcher_demo
```

## 功能演示

### 1. 基本Backend分发
```cpp
// CPU tensor加法
auto cpu_tensor = make_tensor_cpu({2, 3});
callOp("add", {IValue(cpu_tensor), IValue(cpu_tensor)});
// 自动分发到CPU内核实现

// CUDA tensor加法  
auto cuda_tensor = make_tensor_cuda({2, 3});
callOp("add", {IValue(cuda_tensor), IValue(cuda_tensor)});
// 自动分发到CUDA内核实现
```

### 2. 功能性包装器
```cpp
// 启用梯度计算
tensor->setRequiresGrad(true);
callOp("add", {IValue(tensor), IValue(tensor)});
// 分发顺序: Autograd -> CPU

// 启用JIT追踪
GlobalDispatchState::instance().setTracingEnabled(true);
callOp("add", {IValue(tensor), IValue(tensor)});
// 分发顺序: Tracing -> CPU
```

### 3. 组合分发
```cpp
// 多个功能同时启用
tensor->setRequiresGrad(true);
GlobalDispatchState::instance().setTracingEnabled(true);
callOp("add", {IValue(tensor), IValue(tensor)});
// 分发顺序: Autograd -> Tracing -> CPU
```

## 核心算法

### Dispatch算法
1. **计算Dispatch Key Set**: 从输入参数和全局状态收集所有相关的dispatch key
2. **优先级排序**: 按照预定义优先级对dispatch key进行排序  
3. **查找匹配内核**: 按优先级顺序查找第一个已注册的内核函数
4. **执行调用**: 调用找到的内核函数，支持递归分发

### Boxing/Unboxing
- **Boxing**: 将强类型的C++函数参数包装为统一的IValue类型
- **Unboxing**: 从IValue类型还原为强类型参数，调用实际函数
- **类型安全**: 运行时类型检查确保参数类型匹配

## 设计亮点

### 1. 可扩展性
- 新的dispatch key可以轻松添加到枚举中
- 操作符注册完全动态化，支持运行时注册
- 内核函数可以独立注册到不同的dispatch key

### 2. 性能优化
- 使用bitset进行高效的集合操作
- Dispatch key查找时间复杂度为O(k)，k为key数量
- 最小化虚函数调用开销

### 3. 调试支持
- 完整的调试信息输出
- 操作符调用统计和性能监控
- 详细的dispatch过程日志

## 实际应用场景

这个dispatcher设计模式在现代深度学习框架中被广泛应用：

- **PyTorch**: 核心的operator分发机制
- **TensorFlow**: XLA编译器的操作分发
- **JAX**: 设备和变换的分发系统
- **OneFlow**: 分布式执行的任务分发

## 扩展方向

- 添加更多backend支持 (OpenCL, Metal等)
- 实现真正的autograd引擎
- 集成JIT编译器支持
- 添加分布式执行的dispatch key
- 实现内存管理的dispatch机制

## 许可证

MIT License - 详见LICENSE文件 