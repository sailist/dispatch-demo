#include "Dispatcher.h"
#include "TensorImpl.h"
#include <iostream>
#include <cassert>
#include <chrono>

using namespace dispatcher;

// 示例内核函数实现

// CPU加法实现
IValueList add_cpu_kernel(const IValueList& args) {
    std::cout << "  [CPU] 执行加法操作" << std::endl;
    
    if (args.size() != 2 || !args[0].isTensor() || !args[1].isTensor()) {
        throw std::runtime_error("add_cpu_kernel: 需要两个tensor参数");
    }
    
    auto tensor1 = args[0].toTensor();
    auto tensor2 = args[1].toTensor();
    
    std::cout << "    输入1: " << tensor1->debugString() << std::endl;
    std::cout << "    输入2: " << tensor2->debugString() << std::endl;
    
    // 创建结果tensor（简化实现：使用第一个tensor的形状）
    auto result = make_tensor_cpu(tensor1->sizes());
    std::cout << "    输出: " << result->debugString() << std::endl;
    
    return {IValue(result)};
}

// CUDA加法实现
IValueList add_cuda_kernel(const IValueList& args) {
    std::cout << "  [CUDA] 执行加法操作" << std::endl;
    
    if (args.size() != 2 || !args[0].isTensor() || !args[1].isTensor()) {
        throw std::runtime_error("add_cuda_kernel: 需要两个tensor参数");
    }
    
    auto tensor1 = args[0].toTensor();
    auto tensor2 = args[1].toTensor();
    
    std::cout << "    输入1: " << tensor1->debugString() << std::endl;
    std::cout << "    输入2: " << tensor2->debugString() << std::endl;
    
    // 创建结果tensor
    auto result = make_tensor_cuda(tensor1->sizes());
    std::cout << "    输出: " << result->debugString() << std::endl;
    
    return {IValue(result)};
}

// Autograd包装器
IValueList add_autograd_kernel(const IValueList& args) {
    std::cout << "  [Autograd] 包装器：记录梯度信息" << std::endl;
    
    // 在实际实现中，这里会设置梯度计算的forward和backward钩子
    // 为了演示，我们移除autograd key并重新dispatch到backend实现
    
    // 计算当前的dispatch key set
    std::vector<std::shared_ptr<TensorImpl>> tensors;
    for (const auto& arg : args) {
        if (arg.isTensor()) {
            tensors.push_back(arg.toTensor());
        }
    }
    
    auto ks = computeDispatchKeySet(tensors);
    ks.remove(DispatchKey::Autograd);  // 移除autograd key避免无限递归
    
    std::cout << "    重新分发到: " << ks.toString() << std::endl;
    
    // 重新调用dispatcher
    auto result = callOp(OperatorName("add"), ks, args);
    
    std::cout << "    [Autograd] 设置梯度追踪" << std::endl;
    
    return result;
}

// Tracing包装器
IValueList add_tracing_kernel(const IValueList& args) {
    std::cout << "  [Tracing] 包装器：记录操作用于JIT编译" << std::endl;
    
    // 移除tracing key并重新dispatch
    std::vector<std::shared_ptr<TensorImpl>> tensors;
    for (const auto& arg : args) {
        if (arg.isTensor()) {
            tensors.push_back(arg.toTensor());
        }
    }
    
    auto ks = computeDispatchKeySet(tensors);
    ks.remove(DispatchKey::Tracing);
    
    std::cout << "    重新分发到: " << ks.toString() << std::endl;
    
    auto result = callOp(OperatorName("add"), ks, args);
    
    std::cout << "    [Tracing] 记录操作到计算图" << std::endl;
    
    return result;
}

// Profiling包装器
IValueList add_profiling_kernel(const IValueList& args) {
    std::cout << "  [Profiling] 包装器：性能监控开始" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 移除profiling key并重新dispatch
    std::vector<std::shared_ptr<TensorImpl>> tensors;
    for (const auto& arg : args) {
        if (arg.isTensor()) {
            tensors.push_back(arg.toTensor());
        }
    }
    
    auto ks = computeDispatchKeySet(tensors);
    ks.remove(DispatchKey::Profiling);
    
    auto result = callOp(OperatorName("add"), ks, args);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "    [Profiling] 操作耗时: " << duration.count() << " 微秒" << std::endl;
    
    return result;
}

// 注册示例操作符
void registerOperators() {
    std::cout << "=== 注册操作符和内核 ===" << std::endl;
    
    // 注册add操作符
    auto& add_op = REGISTER_OP("add");
    
    // 注册各个dispatch key的内核实现
    REGISTER_KERNEL(add_op, CPU, add_cpu_kernel);
    REGISTER_KERNEL(add_op, CUDA, add_cuda_kernel);
    REGISTER_KERNEL(add_op, Autograd, add_autograd_kernel);
    REGISTER_KERNEL(add_op, Tracing, add_tracing_kernel);
    REGISTER_KERNEL(add_op, Profiling, add_profiling_kernel);
    
    std::cout << "操作符注册完成" << std::endl;
    std::cout << add_op.debugString() << std::endl;
}

// 测试基本dispatch功能
void testBasicDispatch() {
    std::cout << "\n=== 测试基本Dispatch功能 ===" << std::endl;
    
    // 测试CPU tensors
    std::cout << "\n1. CPU tensor加法:" << std::endl;
    auto cpu_tensor1 = make_tensor_cpu({2, 3});
    auto cpu_tensor2 = make_tensor_cpu({2, 3});
    
    IValueList cpu_args = {IValue(cpu_tensor1), IValue(cpu_tensor2)};
    auto cpu_result = callOp("add", cpu_args);
    
    // 测试CUDA tensors
    std::cout << "\n2. CUDA tensor加法:" << std::endl;
    auto cuda_tensor1 = make_tensor_cuda({3, 4});
    auto cuda_tensor2 = make_tensor_cuda({3, 4});
    
    IValueList cuda_args = {IValue(cuda_tensor1), IValue(cuda_tensor2)};
    auto cuda_result = callOp("add", cuda_args);
}

// 测试功能性dispatch key
void testFunctionalityKeys() {
    std::cout << "\n=== 测试功能性Dispatch Keys ===" << std::endl;
    
    // 创建需要梯度的tensor
    std::cout << "\n1. 启用Autograd的tensor加法:" << std::endl;
    auto tensor1 = make_tensor_cpu({2, 2});
    auto tensor2 = make_tensor_cpu({2, 2});
    tensor1->setRequiresGrad(true);  // 启用梯度计算
    
    IValueList args = {IValue(tensor1), IValue(tensor2)};
    auto result = callOp("add", args);
    
    // 测试全局tracing状态
    std::cout << "\n2. 启用全局Tracing状态:" << std::endl;
    GlobalDispatchState::instance().setTracingEnabled(true);
    
    auto tensor3 = make_tensor_cpu({1, 4});
    auto tensor4 = make_tensor_cpu({1, 4});
    
    IValueList tracing_args = {IValue(tensor3), IValue(tensor4)};
    auto tracing_result = callOp("add", tracing_args);
    
    GlobalDispatchState::instance().setTracingEnabled(false);  // 关闭tracing
    
    // 测试全局profiling状态
    std::cout << "\n3. 启用全局Profiling状态:" << std::endl;
    GlobalDispatchState::instance().setProfilingEnabled(true);
    
    auto tensor5 = make_tensor_cpu({3, 3});
    auto tensor6 = make_tensor_cpu({3, 3});
    
    IValueList profiling_args = {IValue(tensor5), IValue(tensor6)};
    auto profiling_result = callOp("add", profiling_args);
    
    GlobalDispatchState::instance().setProfilingEnabled(false);  // 关闭profiling
}

// 测试组合dispatch keys
void testCombinedKeys() {
    std::cout << "\n=== 测试组合Dispatch Keys ===" << std::endl;
    
    std::cout << "\n1. Autograd + Tracing + CPU:" << std::endl;
    GlobalDispatchState::instance().setTracingEnabled(true);
    
    auto tensor1 = make_tensor_cpu({2, 2});
    auto tensor2 = make_tensor_cpu({2, 2});
    tensor1->setRequiresGrad(true);
    
    IValueList args = {IValue(tensor1), IValue(tensor2)};
    
    std::cout << "Dispatch key set: " << tensor1->keySet().toString() << std::endl;
    auto result = callOp("add", args);
    
    GlobalDispatchState::instance().setTracingEnabled(false);
    
    std::cout << "\n2. 所有功能性keys + CUDA:" << std::endl;
    GlobalDispatchState::instance().setAutogradEnabled(true);
    GlobalDispatchState::instance().setTracingEnabled(true);
    GlobalDispatchState::instance().setProfilingEnabled(true);
    
    auto cuda_tensor1 = make_tensor_cuda({1, 2});
    auto cuda_tensor2 = make_tensor_cuda({1, 2});
    
    IValueList cuda_args = {IValue(cuda_tensor1), IValue(cuda_tensor2)};
    
    std::cout << "全局dispatch keys: " << 
        GlobalDispatchState::instance().computeFunctionalityKeys().toString() << std::endl;
    
    auto cuda_result = callOp("add", cuda_args);
    
    // 重置全局状态
    GlobalDispatchState::instance().setAutogradEnabled(false);
    GlobalDispatchState::instance().setTracingEnabled(false);
    GlobalDispatchState::instance().setProfilingEnabled(false);
}

// 测试性能统计
void testProfiling() {
    std::cout << "\n=== 测试性能统计功能 ===" << std::endl;
    
    Dispatcher::instance().enableProfiling(true);
    
    // 进行多次调用
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n调用 #" << (i+1) << ":" << std::endl;
        
        auto tensor1 = make_tensor_cpu({i+1, i+1});
        auto tensor2 = make_tensor_cpu({i+1, i+1});
        
        IValueList args = {IValue(tensor1), IValue(tensor2)};
        callOp("add", args);
    }
    
    // 打印统计信息
    std::cout << "\n=== 性能统计报告 ===" << std::endl;
    Dispatcher::instance().printDebugInfo();
    
    Dispatcher::instance().enableProfiling(false);
}

int main() {
    try {
        std::cout << "PyTorch风格Dispatcher演示程序" << std::endl;
        std::cout << "================================" << std::endl;
        
        // 注册操作符
        registerOperators();
        
        // 测试基本功能
        testBasicDispatch();
        
        // 测试功能性dispatch keys
        testFunctionalityKeys();
        
        // 测试组合keys
        testCombinedKeys();
        
        // 测试性能统计
        testProfiling();
        
        std::cout << "\n=== 最终Dispatcher状态 ===" << std::endl;
        Dispatcher::instance().printDebugInfo();
        
        std::cout << "\n程序执行完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 