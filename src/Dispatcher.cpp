#include "Dispatcher.h"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <mutex>

namespace dispatcher {

// Dispatcher实现
Dispatcher& Dispatcher::instance() {
    static Dispatcher instance;
    return instance;
}

OperatorHandle& Dispatcher::registerOperator(const OperatorName& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    // 检查操作符是否已经注册
    auto it = operators_.find(name);
    if (it != operators_.end()) {
        return *it->second;
    }
    
    // 创建新的操作符句柄
    auto handle = std::make_unique<OperatorHandle>(name.fullName());
    OperatorHandle& handle_ref = *handle;
    
    // 插入到注册表中
    operators_[name] = std::move(handle);
    
    // 通知回调函数
    notifyRegistrationCallbacks(name, true);
    
    return handle_ref;
}

OperatorHandle* Dispatcher::findOperator(const OperatorName& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = operators_.find(name);
    return (it != operators_.end()) ? it->second.get() : nullptr;
}

const OperatorHandle* Dispatcher::findOperator(const OperatorName& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = operators_.find(name);
    return (it != operators_.end()) ? it->second.get() : nullptr;
}

void Dispatcher::deregisterOperator(const OperatorName& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = operators_.find(name);
    if (it != operators_.end()) {
        operators_.erase(it);
        // 通知回调函数
        notifyRegistrationCallbacks(name, false);
    }
}

bool Dispatcher::hasOperator(const OperatorName& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return operators_.find(name) != operators_.end();
}

std::vector<OperatorName> Dispatcher::getAllOperatorNames() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<OperatorName> names;
    for (const auto& entry : operators_) {
        names.push_back(entry.first);
    }
    
    return names;
}

IValueList Dispatcher::call(const OperatorName& name, const IValueList& args) const {
    // 查找操作符
    const OperatorHandle* handle = findOperator(name);
    if (!handle) {
        throw std::runtime_error("Operator '" + name.fullName() + "' is not registered");
    }
    
    // 调用操作符（让OperatorHandle计算dispatch key set）
    auto result = handle->call(args);
    
    // 更新统计信息
    if (profiling_enabled_) {
        // 从参数计算dispatch key set以获取使用的key
        auto ks = handle->computeDispatchKeySet(args);
        auto key = ks.highestPriorityKey();
        updateCallStats(name, key);
    }
    
    return result;
}

IValueList Dispatcher::call(const OperatorName& name, const DispatchKeySet& ks, const IValueList& args) const {
    // 查找操作符
    const OperatorHandle* handle = findOperator(name);
    if (!handle) {
        throw std::runtime_error("Operator '" + name.fullName() + "' is not registered");
    }
    
    // 使用指定的dispatch key set调用操作符
    auto result = handle->call(ks, args);
    
    // 更新统计信息
    if (profiling_enabled_) {
        updateCallStats(name, ks.highestPriorityKey());
    }
    
    return result;
}

IValueList Dispatcher::call(const std::string& name, const IValueList& args) const {
    return call(OperatorName(name), args);
}

std::string Dispatcher::debugString() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::ostringstream oss;
    oss << "Dispatcher {\n";
    oss << "  Registered operators: " << operators_.size() << "\n";
    
    for (const auto& entry : operators_) {
        oss << "  " << entry.first.fullName() << " {\n";
        auto keys = entry.second->getRegisteredKeys();
        for (auto key : keys) {
            oss << "    " << toString(key) << "\n";
        }
        oss << "  }\n";
    }
    
    if (profiling_enabled_) {
        oss << "\n  Call Statistics:\n";
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        for (const auto& stat_entry : call_stats_) {
            oss << "    " << stat_entry.first.fullName() 
                << ": " << stat_entry.second.call_count << " calls\n";
            for (const auto& key_count : stat_entry.second.key_counts) {
                oss << "      " << toString(key_count.first) 
                    << ": " << key_count.second << " times\n";
            }
        }
    }
    
    oss << "}";
    return oss.str();
}

void Dispatcher::printDebugInfo() const {
    std::cout << debugString() << std::endl;
}

void Dispatcher::addRegistrationCallback(OperatorRegistrationCallback callback) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    registration_callbacks_.push_back(std::move(callback));
}

void Dispatcher::resetCallStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    call_stats_.clear();
}

void Dispatcher::notifyRegistrationCallbacks(const OperatorName& name, bool registered) {
    // 注意：这个函数在持有锁的情况下被调用
    for (const auto& callback : registration_callbacks_) {
        try {
            callback(name, registered);
        } catch (...) {
            // 忽略回调函数中的异常
        }
    }
}

void Dispatcher::updateCallStats(const OperatorName& name, DispatchKey key) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto& stats = call_stats_[name];
    stats.call_count++;
    stats.key_counts[key]++;
}

// 全局便捷函数实现
OperatorHandle& registerOp(const OperatorName& name) {
    return Dispatcher::instance().registerOperator(name);
}

OperatorHandle& registerOp(const std::string& name) {
    return registerOp(OperatorName(name));
}

OperatorHandle& registerOp(const std::string& name, const std::string& overload) {
    return registerOp(OperatorName(name, overload));
}

IValueList callOp(const OperatorName& name, const IValueList& args) {
    return Dispatcher::instance().call(name, args);
}

IValueList callOp(const OperatorName& name, const DispatchKeySet& ks, const IValueList& args) {
    return Dispatcher::instance().call(name, ks, args);
}

IValueList callOp(const std::string& name, const IValueList& args) {
    return Dispatcher::instance().call(name, args);
}

} // namespace dispatcher 