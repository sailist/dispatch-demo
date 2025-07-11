#pragma once

#include "OperatorHandle.h"
#include "DispatchKey.h"
#include "DispatchKeySet.h"
#include "IValue.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

namespace dispatcher {

// OperatorName - 操作符名称和重载标识符
struct OperatorName {
    std::string name;           // 操作符基础名称，如 "add"
    std::string overload_name;  // 重载名称，如 "tensor" 或 ""（默认重载）
    
    OperatorName(std::string name) : name(std::move(name)) {}
    OperatorName(std::string name, std::string overload) 
        : name(std::move(name)), overload_name(std::move(overload)) {}
    
    // 获取完整的操作符名称 "name.overload" 或 "name"
    std::string fullName() const {
        return overload_name.empty() ? name : name + "." + overload_name;
    }
    
    // 比较操作符，用于map中的键
    bool operator==(const OperatorName& other) const {
        return name == other.name && overload_name == other.overload_name;
    }
    
    bool operator<(const OperatorName& other) const {
        if (name != other.name) return name < other.name;
        return overload_name < other.overload_name;
    }
};

} // namespace dispatcher

// 为OperatorName提供hash函数，用于unordered_map
namespace std {
template<>
struct hash<dispatcher::OperatorName> {
    size_t operator()(const dispatcher::OperatorName& op) const {
        return hash<string>()(op.name) ^ (hash<string>()(op.overload_name) << 1);
    }
};
}

namespace dispatcher {

// Dispatcher - 全局操作符分发器
// 这是整个系统的核心，维护所有操作符的注册表
class Dispatcher {
public:
    // 获取全局单例
    static Dispatcher& instance();
    
    // 析构函数
    ~Dispatcher() = default;
    
    // 注册新的操作符 - 返回操作符句柄用于后续内核注册
    OperatorHandle& registerOperator(const OperatorName& name);
    
    // 查找已注册的操作符
    OperatorHandle* findOperator(const OperatorName& name);
    const OperatorHandle* findOperator(const OperatorName& name) const;
    
    // 注销操作符
    void deregisterOperator(const OperatorName& name);
    
    // 检查操作符是否已注册
    bool hasOperator(const OperatorName& name) const;
    
    // 获取所有已注册操作符的名称列表
    std::vector<OperatorName> getAllOperatorNames() const;
    
    // 调用操作符 - 这是外部调用的主要接口
    IValueList call(const OperatorName& name, const IValueList& args) const;
    IValueList call(const OperatorName& name, const DispatchKeySet& ks, const IValueList& args) const;
    
    // 便捷调用接口 - 直接使用字符串名称
    IValueList call(const std::string& name, const IValueList& args) const;
    
    // 调试功能 - 打印所有注册的操作符和内核
    std::string debugString() const;
    void printDebugInfo() const;
    
    // 回调系统 - 用于监控操作符注册/注销
    using OperatorRegistrationCallback = std::function<void(const OperatorName&, bool /*registered*/)>;
    void addRegistrationCallback(OperatorRegistrationCallback callback);
    
    // 性能监控 - 统计每个操作符的调用次数
    struct CallStats {
        size_t call_count = 0;
        std::unordered_map<DispatchKey, size_t> key_counts;
    };
    
    void enableProfiling(bool enabled) { profiling_enabled_ = enabled; }
    bool isProfilingEnabled() const { return profiling_enabled_; }
    const std::unordered_map<OperatorName, CallStats>& getCallStats() const { return call_stats_; }
    void resetCallStats();

private:
    // 私有构造函数 - 单例模式
    Dispatcher() = default;
    
    // 禁用拷贝和赋值
    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;
    
    // 操作符注册表 - 操作符名称到句柄的映射
    std::unordered_map<OperatorName, std::unique_ptr<OperatorHandle>> operators_;
    
    // 线程安全 - 保护注册表的并发访问
    mutable std::mutex registry_mutex_;
    
    // 回调函数列表
    std::vector<OperatorRegistrationCallback> registration_callbacks_;
    
    // 性能统计
    bool profiling_enabled_ = false;
    mutable std::unordered_map<OperatorName, CallStats> call_stats_;
    mutable std::mutex stats_mutex_;
    
    // 内部辅助函数
    void notifyRegistrationCallbacks(const OperatorName& name, bool registered);
    void updateCallStats(const OperatorName& name, DispatchKey key) const;
};

// 全局便捷函数 - 直接访问默认dispatcher
OperatorHandle& registerOp(const OperatorName& name);
OperatorHandle& registerOp(const std::string& name);
OperatorHandle& registerOp(const std::string& name, const std::string& overload);

IValueList callOp(const OperatorName& name, const IValueList& args);
IValueList callOp(const OperatorName& name, const DispatchKeySet& ks, const IValueList& args);
IValueList callOp(const std::string& name, const IValueList& args);

// 便捷宏 - 简化操作符注册
#define REGISTER_OP(name) \
    dispatcher::registerOp(name)

#define REGISTER_OP_OVERLOAD(name, overload) \
    dispatcher::registerOp(name, overload)

} // namespace dispatcher 