#pragma once

#include "DispatchKey.h"
#include "DispatchKeySet.h"
#include "IValue.h"
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>

namespace dispatcher {

// 前向声明
class TensorImpl;

// 函数指针类型定义
// Boxed函数：接受和返回IValue列表
using BoxedKernelFunction = std::function<IValueList(const IValueList&)>;

// Unboxed函数：直接的C++函数类型（这里简化为void*，实际中需要模板特化）
using UnboxedKernelFunction = void*;

// KernelFunction - 封装boxed和unboxed函数
class KernelFunction {
public:
    // 默认构造函数 - 创建无效的函数
    KernelFunction() = default;
    
    // 构造函数 - 从boxed函数创建
    explicit KernelFunction(BoxedKernelFunction boxed_fn);
    
    // 构造函数 - 从unboxed函数创建（需要boxing wrapper）
    template<typename Func>
    explicit KernelFunction(Func&& unboxed_fn);
    
    // 调用boxed函数
    IValueList callBoxed(const IValueList& args) const;
    
    // 检查是否有效
    bool isValid() const { return static_cast<bool>(boxed_fn_); }

private:
    BoxedKernelFunction boxed_fn_;
    
    // 内部辅助 - 将unboxed函数包装为boxed函数
    template<typename Func>
    BoxedKernelFunction makeBoxedFromUnboxed(Func&& unboxed_fn);
};

// OperatorHandle - 管理单个操作符的dispatch table
class OperatorHandle {
public:
    // 构造函数
    explicit OperatorHandle(std::string name);
    
    // 析构函数
    ~OperatorHandle() = default;
    
    // 获取操作符名称
    const std::string& name() const { return name_; }
    
    // 注册内核函数到指定的dispatch key
    void setKernel(DispatchKey key, KernelFunction kernel);
    
    // 移除指定dispatch key的内核函数
    void removeKernel(DispatchKey key);
    
    // 检查是否有指定dispatch key的内核函数
    bool hasKernel(DispatchKey key) const;
    
    // 根据dispatch key set查找最佳匹配的内核函数
    // 这是dispatch的核心逻辑：按优先级顺序查找第一个可用的内核
    const KernelFunction* findKernel(const DispatchKeySet& ks) const;
    
    // 调用操作符 - 根据dispatch key set自动选择合适的内核
    IValueList call(const DispatchKeySet& ks, const IValueList& args) const;
    
    // 便捷调用接口 - 从tensor参数自动计算dispatch key set
    IValueList call(const IValueList& args) const;
    
    // 获取所有已注册的dispatch key
    std::vector<DispatchKey> getRegisteredKeys() const;
    
    // 调试信息 - 显示dispatch table的内容
    std::string debugString() const;
    
    // 从IValue参数计算dispatch key set（公有方法，供Dispatcher使用）
    DispatchKeySet computeDispatchKeySet(const IValueList& args) const;

private:
    std::string name_;
    
    // Dispatch table: dispatch key -> 内核函数
    // 使用unordered_map存储每个dispatch key对应的函数实现
    std::unordered_map<DispatchKey, KernelFunction> dispatch_table_;
};

// 便捷宏定义 - 用于简化内核函数注册
#define REGISTER_KERNEL(op_handle, dispatch_key, func) \
    (op_handle).setKernel(DispatchKey::dispatch_key, KernelFunction(func))

// 模板实现 - KernelFunction的unboxed构造函数
template<typename Func>
KernelFunction::KernelFunction(Func&& unboxed_fn) 
    : boxed_fn_(makeBoxedFromUnboxed(std::forward<Func>(unboxed_fn))) {
}

// 模板实现 - 将unboxed函数包装为boxed函数
template<typename Func>
BoxedKernelFunction KernelFunction::makeBoxedFromUnboxed(Func&& unboxed_fn) {
    return [unboxed_fn = std::forward<Func>(unboxed_fn)](const IValueList& args) -> IValueList {
        // 这里是boxing/unboxing的核心逻辑
        // 实际实现中需要根据函数签名进行类型转换
        // 为了简化，这里假设所有unboxed函数都接受和返回特定类型
        
        // 示例：假设函数签名为 tensor add(tensor, tensor)
        if (args.size() >= 2 && args[0].isTensor() && args[1].isTensor()) {
            auto tensor1 = args[0].toTensor();
            auto tensor2 = args[1].toTensor();
            
            // 调用实际的unboxed函数（这里需要类型安全的转换）
            // auto result = unboxed_fn(tensor1, tensor2);
            // return {IValue(result)};
            
            // 简化处理：直接返回第一个tensor
            return {args[0]};
        }
        
        // 默认情况：返回空列表
        return {};
    };
}

} // namespace dispatcher 