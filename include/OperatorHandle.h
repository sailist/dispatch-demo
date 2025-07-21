#pragma once

#include "DispatchKey.h"
#include "DispatchKeySet.h"
#include "IValue.h"
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <tuple>

namespace dispatcher {

// 前向声明
class TensorImpl;

// === Boxing/Unboxing 类型萃取系统 ===

// 类型别名 - 简化 Tensor 类型
using Tensor = std::shared_ptr<TensorImpl>;

// 类型萃取 - 将 C++ 类型转换为 IValue
template<typename T>
struct arg_to_ivalue {
    static IValue convert(const T& value) {
        return IValue(value);
    }
};

// 特化 - Tensor 类型
template<>
struct arg_to_ivalue<Tensor> {
    static IValue convert(const Tensor& tensor) {
        return IValue(tensor);
    }
};

// 类型萃取 - 从 IValue 提取 C++ 类型
template<typename T>
struct ivalue_to_arg {
    static T convert(const IValue& ivalue);
};

// 特化 - Tensor 类型
template<>
struct ivalue_to_arg<Tensor> {
    static Tensor convert(const IValue& ivalue) {
        if (!ivalue.isTensor()) {
            throw std::runtime_error("Expected Tensor type");
        }
        return ivalue.toTensor();
    }
};

// 特化 - const Tensor& 类型
template<>
struct ivalue_to_arg<const Tensor&> {
    static const Tensor& convert(const IValue& ivalue) {
        if (!ivalue.isTensor()) {
            throw std::runtime_error("Expected Tensor type");
        }
        // 注意：这里返回临时对象的引用是不安全的，但为了演示简化处理
        // 在实际实现中需要更复杂的生命周期管理
        static thread_local Tensor temp_tensor;
        temp_tensor = ivalue.toTensor();
        return temp_tensor;
    }
};

// 特化 - Tensor& 类型
template<>
struct ivalue_to_arg<Tensor&> {
    static Tensor& convert(const IValue& ivalue) {
        if (!ivalue.isTensor()) {
            throw std::runtime_error("Expected Tensor type");
        }
        static thread_local Tensor temp_tensor;
        temp_tensor = ivalue.toTensor();
        return temp_tensor;
    }
};

// 特化 - double 类型
template<>
struct ivalue_to_arg<double> {
    static double convert(const IValue& ivalue) {
        if (!ivalue.isDouble()) {
            throw std::runtime_error("Expected Double type");
        }
        return ivalue.toDouble();
    }
};

// 特化 - const double& 类型
template<>
struct ivalue_to_arg<const double&> {
    static const double& convert(const IValue& ivalue) {
        if (!ivalue.isDouble()) {
            throw std::runtime_error("Expected Double type");
        }
        static thread_local double temp_double;
        temp_double = ivalue.toDouble();
        return temp_double;
    }
};

// 特化 - int64_t 类型
template<>
struct ivalue_to_arg<int64_t> {
    static int64_t convert(const IValue& ivalue) {
        if (!ivalue.isInt()) {
            throw std::runtime_error("Expected Int type");
        }
        return ivalue.toInt();
    }
};

// 特化 - const int64_t& 类型
template<>
struct ivalue_to_arg<const int64_t&> {
    static const int64_t& convert(const IValue& ivalue) {
        if (!ivalue.isInt()) {
            throw std::runtime_error("Expected Int type");
        }
        static thread_local int64_t temp_int;
        temp_int = ivalue.toInt();
        return temp_int;
    }
};

// 特化 - bool 类型
template<>
struct ivalue_to_arg<bool> {
    static bool convert(const IValue& ivalue) {
        if (!ivalue.isBool()) {
            throw std::runtime_error("Expected Bool type");
        }
        return ivalue.toBool();
    }
};

// 特化 - const bool& 类型
template<>
struct ivalue_to_arg<const bool&> {
    static const bool& convert(const IValue& ivalue) {
        if (!ivalue.isBool()) {
            throw std::runtime_error("Expected Bool type");
        }
        static thread_local bool temp_bool;
        temp_bool = ivalue.toBool();
        return temp_bool;
    }
};

// 函数类型萃取 - 提取函数签名信息
template<typename F>
struct function_traits;

template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using return_type = R;
    using arg_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using arg_type = typename std::tuple_element<N, arg_types>::type;
};

template<typename R, typename... Args>
struct function_traits<R(&)(Args...)> {
    using return_type = R;
    using arg_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using arg_type = typename std::tuple_element<N, arg_types>::type;
};

template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
    using return_type = R;
    using arg_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using arg_type = typename std::tuple_element<N, arg_types>::type;
};

// Lambda 和函数对象支持
template<typename F>
struct function_traits : function_traits<decltype(&F::operator())> {};

template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    using arg_types = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
    
    template<size_t N>
    using arg_type = typename std::tuple_element<N, arg_types>::type;
};

// === 函数类型检测 ===

// 检测是否是 boxed 函数类型
template<typename F>
struct is_boxed_function {
    static constexpr bool value = false;
};

// 特化 - 检测 IValueList(const IValueList&) 类型的函数指针
template<>
struct is_boxed_function<IValueList(*)(const IValueList&)> {
    static constexpr bool value = true;
};

// 特化 - 检测 IValueList(const IValueList&) 类型的函数引用
template<>
struct is_boxed_function<IValueList(&)(const IValueList&)> {
    static constexpr bool value = true;
};

// 特化 - 检测 std::function<IValueList(const IValueList&)> 类型
template<>
struct is_boxed_function<std::function<IValueList(const IValueList&)>> {
    static constexpr bool value = true;
};

// 对于 function pointer 类型的检测
template<typename F>
constexpr bool is_boxed_function_v = is_boxed_function<F>::value;

// === Boxing/Unboxing 辅助函数 ===

// 静态函数 - 包装返回值
template<typename R>
IValueList wrapReturn(R&& result) {
    if constexpr (std::is_void_v<std::decay_t<R>>) {
        return {};
    } else {
        return {arg_to_ivalue<std::decay_t<R>>::convert(std::forward<R>(result))};
    }
}

// 静态函数 - 调用 unboxed 函数并包装返回值
template<typename Func, size_t... Is>
IValueList callUnboxedImpl(Func&& func, const IValueList& args, std::index_sequence<Is...>) {
    using traits = function_traits<std::decay_t<Func>>;
    using return_type = typename traits::return_type;
    
    // 从IValue列表中提取每个参数，转换为正确的C++类型，然后调用函数
    if constexpr (std::is_void_v<return_type>) {
        // void 返回类型
        func(ivalue_to_arg<typename traits::template arg_type<Is>>::convert(args[Is])...);
        return {};
    } else {
        // 有返回值的类型
        auto result = func(ivalue_to_arg<typename traits::template arg_type<Is>>::convert(args[Is])...);
        return wrapReturn(std::move(result));
    }
}

// === KernelFunction 和相关类型定义 ===

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
    
    // 构造函数 - 智能构造：自动检测函数类型并决定是否需要boxing
    template<typename Func>
    explicit KernelFunction(Func&& func);
    
    // 调用boxed函数
    IValueList callBoxed(const IValueList& args) const;
    
    // 检查是否有效
    bool isValid() const { return static_cast<bool>(boxed_fn_); }

private:
    BoxedKernelFunction boxed_fn_;
    
    // 辅助函数 - 将unboxed函数包装为boxed函数
    template<typename Func>
    static BoxedKernelFunction makeBoxedFromUnboxed(Func&& unboxed_fn);
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

// === 模板实现部分 ===

// KernelFunction的智能构造函数 - 自动检测函数类型
template<typename Func>
KernelFunction::KernelFunction(Func&& func) {
    using DecayedFunc = std::decay_t<Func>;
    
    // 检查是否是已经 boxed 的函数
    if constexpr (is_boxed_function_v<DecayedFunc>) {
        // 已经是 boxed 函数，直接使用
        boxed_fn_ = BoxedKernelFunction(std::forward<Func>(func));
    } else {
        // 是 unboxed 函数，需要进行 boxing
        boxed_fn_ = makeBoxedFromUnboxed(std::forward<Func>(func));
    }
}

// 核心实现 - 将unboxed函数包装为boxed函数
template<typename Func>
BoxedKernelFunction KernelFunction::makeBoxedFromUnboxed(Func&& unboxed_fn) {
    using traits = function_traits<std::decay_t<Func>>;
    
    return [unboxed_fn = std::forward<Func>(unboxed_fn)](const IValueList& args) -> IValueList {
        // 第一步：验证参数数量
        if (args.size() != traits::arity) {
            throw std::runtime_error("参数数量不匹配：期望 " + std::to_string(traits::arity) + 
                                   " 个参数，实际得到 " + std::to_string(args.size()) + " 个");
        }
        
        // 第二步：使用index_sequence展开参数并调用unboxed函数
        return callUnboxedImpl(unboxed_fn, args, std::make_index_sequence<traits::arity>{});
    };
}

} // namespace dispatcher 