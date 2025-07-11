# PyTorch-Style Dispatcher C++ Implementation

This is a demo project for learning PyTorch dispatch mechanisms based on the [PyTorch Dispatcher paper](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/).

## Project Features

- 🚀 **Multiple Dispatch**: Supports function dispatch based on multiple dispatch keys
- 📦 **Boxing/Unboxing**: Unified function calling convention support
- 🔧 **Backend Abstraction**: Transparent switching between CPU, CUDA and other backends
- 🧩 **Functional Wrappers**: Cross-cutting concerns like Autograd, Tracing, Profiling
- 📊 **Performance Monitoring**: Built-in operator call statistics
- 🧵 **Thread Safety**: Supports concurrent access in multi-threaded environments

## Core Concepts

### Dispatch Key
```cpp
enum class DispatchKey {
    CPU,        // CPU backend implementation
    CUDA,       // CUDA backend implementation
    Autograd,   // Automatic differentiation wrapper
    Tracing,    // JIT tracing wrapper
    Profiling,  // Performance monitoring wrapper
    CatchAll    // Fallback implementation
};
```

### Dispatch Key Set
Uses bitset for efficient dispatch key set management with priority sorting:
- Functional keys (Autograd, Tracing, Profiling) have higher priority
- Backend keys (CPU, CUDA) provide concrete implementations

### Boxing Mechanism
```cpp
// Unified IValue type supports arbitrary parameter types
IValue cpu_tensor = IValue(make_tensor_cpu({2, 3}));
IValue result = callOp("add", {cpu_tensor, cpu_tensor});
```

## Project Structure

```
dispatcher-demo/
├── CMakeLists.txt          # CMake build configuration
├── build.sh               # Build script
├── README.md              # Project documentation (Chinese)
├── README_EN.md           # Project documentation (English)
├── include/               # Header files directory
│   ├── DispatchKey.h      # Dispatch key enumeration definition
│   ├── DispatchKeySet.h   # Dispatch key set management
│   ├── IValue.h           # Boxing/Unboxing mechanism
│   ├── TensorImpl.h       # Simplified Tensor implementation
│   ├── OperatorHandle.h   # Operator handle and function table
│   └── Dispatcher.h       # Core dispatcher
└── src/                   # Source files directory
    ├── DispatchKeySet.cpp # Key set implementation
    ├── IValue.cpp         # IValue class implementation
    ├── TensorImpl.cpp     # Tensor implementation
    ├── OperatorHandle.cpp # Operator handle implementation
    ├── Dispatcher.cpp     # Dispatcher core logic
    └── main.cpp           # Demo program
```

## Build Instructions

### Requirements
- CMake 3.15+
- Clang compiler (with C++17 support)
- Ninja build system

### Build Steps
```bash
# Grant execution permission to build script
chmod +x build.sh

# Run build script
./build.sh

# Run demo program
cd build && ./bin/dispatcher_demo
```

### Manual Build
```bash
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_CXX_COMPILER=clang++
ninja
./bin/dispatcher_demo
```

## Feature Demonstrations

### 1. Basic Backend Dispatch
```cpp
// CPU tensor addition
auto cpu_tensor = make_tensor_cpu({2, 3});
callOp("add", {IValue(cpu_tensor), IValue(cpu_tensor)});
// Automatically dispatches to CPU kernel implementation

// CUDA tensor addition
auto cuda_tensor = make_tensor_cuda({2, 3});
callOp("add", {IValue(cuda_tensor), IValue(cuda_tensor)});
// Automatically dispatches to CUDA kernel implementation
```

### 2. Functional Wrappers
```cpp
// Enable gradient computation
tensor->setRequiresGrad(true);
callOp("add", {IValue(tensor), IValue(tensor)});
// Dispatch order: Autograd -> CPU

// Enable JIT tracing
GlobalDispatchState::instance().setTracingEnabled(true);
callOp("add", {IValue(tensor), IValue(tensor)});
// Dispatch order: Tracing -> CPU
```

### 3. Combined Dispatch
```cpp
// Multiple features enabled simultaneously
tensor->setRequiresGrad(true);
GlobalDispatchState::instance().setTracingEnabled(true);
callOp("add", {IValue(tensor), IValue(tensor)});
// Dispatch order: Autograd -> Tracing -> CPU
```

## Core Algorithms

### Dispatch Algorithm
1. **Calculate Dispatch Key Set**: Collect all relevant dispatch keys from input parameters and global state
2. **Priority Sorting**: Sort dispatch keys according to predefined priority order
3. **Find Matching Kernel**: Find the first registered kernel function in priority order
4. **Execute Call**: Call the found kernel function, supporting recursive dispatch

### Boxing/Unboxing
- **Boxing**: Wrap strongly-typed C++ function parameters into unified IValue type
- **Unboxing**: Restore IValue type back to strongly-typed parameters, call actual function
- **Type Safety**: Runtime type checking ensures parameter type matching

## Design Highlights

### 1. Extensibility
- New dispatch keys can be easily added to the enumeration
- Operator registration is completely dynamic, supporting runtime registration
- Kernel functions can be independently registered to different dispatch keys

### 2. Performance Optimization
- Uses bitset for efficient set operations
- Dispatch key lookup time complexity is O(k), where k is the number of keys
- Minimizes virtual function call overhead

### 3. Debugging Support
- Complete debug information output
- Operator call statistics and performance monitoring
- Detailed dispatch process logging

## Real-world Use Cases

This dispatcher design pattern is widely used in modern deep learning frameworks:

- **PyTorch**: Core operator dispatch mechanism
- **TensorFlow**: XLA compiler operation dispatch
- **JAX**: Device and transformation dispatch system
- **OneFlow**: Task dispatch for distributed execution

## Extension Directions

- Add support for more backends (OpenCL, Metal, etc.)
- Implement real autograd engine
- Integrate JIT compiler support
- Add dispatch keys for distributed execution
- Implement memory management dispatch mechanism

## License

MIT License - See LICENSE file for details 