#include "DispatchKeySet.h"
#include "DispatchKey.h"
#include <sstream>
#include <algorithm>

namespace dispatcher {

// DispatchKey辅助函数实现
const char* toString(DispatchKey key) {
    switch (key) {
        case DispatchKey::CPU: return "CPU";
        case DispatchKey::CUDA: return "CUDA";
        case DispatchKey::Autograd: return "Autograd";
        case DispatchKey::Tracing: return "Tracing";
        case DispatchKey::Profiling: return "Profiling";
        case DispatchKey::Undefined: return "Undefined";
        case DispatchKey::CatchAll: return "CatchAll";
        default: return "Unknown";
    }
}

uint8_t dispatchKeyPriority(DispatchKey key) {
    // 优先级定义：数字越小优先级越高
    // Functionality keys优先于Backend keys，这样功能性包装器可以先执行
    switch (key) {
        case DispatchKey::Autograd: return 0;    // 最高优先级 - autograd需要包装所有操作
        case DispatchKey::Tracing: return 1;     // 第二优先级 - tracing包装
        case DispatchKey::Profiling: return 2;   // 第三优先级 - profiling包装
        case DispatchKey::CPU: return 10;        // Backend实现
        case DispatchKey::CUDA: return 11;       // Backend实现
        case DispatchKey::CatchAll: return 100;  // 最低优先级 - 兜底实现
        case DispatchKey::Undefined: return 255; // 未定义，不应该被选中
        default: return 128;
    }
}

bool isBackendKey(DispatchKey key) {
    return key == DispatchKey::CPU || key == DispatchKey::CUDA;
}

bool isFunctionalityKey(DispatchKey key) {
    return key == DispatchKey::Autograd || key == DispatchKey::Tracing || key == DispatchKey::Profiling;
}

// DispatchKeySet实现
DispatchKeySet::DispatchKeySet(DispatchKey key) {
    add(key);
}

DispatchKeySet::DispatchKeySet(std::initializer_list<DispatchKey> keys) {
    for (auto key : keys) {
        add(key);
    }
}

DispatchKeySet& DispatchKeySet::add(DispatchKey key) {
    raw_repr_.set(keyIndex(key));
    return *this;
}

DispatchKeySet& DispatchKeySet::remove(DispatchKey key) {
    raw_repr_.reset(keyIndex(key));
    return *this;
}

bool DispatchKeySet::has(DispatchKey key) const {
    return raw_repr_.test(keyIndex(key));
}

bool DispatchKeySet::empty() const {
    return raw_repr_.none();
}

void DispatchKeySet::clear() {
    raw_repr_.reset();
}

DispatchKey DispatchKeySet::highestPriorityKey() const {
    if (empty()) {
        return DispatchKey::Undefined;
    }
    
    // 创建优先级排序的dispatch key列表
    std::vector<DispatchKey> keys_by_priority;
    for (int i = 0; i < static_cast<int>(DispatchKey::NumDispatchKeys); ++i) {
        auto key = static_cast<DispatchKey>(i);
        if (has(key)) {
            keys_by_priority.push_back(key);
        }
    }
    
    // 按优先级排序（优先级数字小的在前）
    std::sort(keys_by_priority.begin(), keys_by_priority.end(),
              [](DispatchKey a, DispatchKey b) {
                  return dispatchKeyPriority(a) < dispatchKeyPriority(b);
              });
    
    return keys_by_priority.empty() ? DispatchKey::Undefined : keys_by_priority[0];
}

DispatchKeySet DispatchKeySet::operator|(const DispatchKeySet& other) const {
    DispatchKeySet result = *this;
    result |= other;
    return result;
}

DispatchKeySet& DispatchKeySet::operator|=(const DispatchKeySet& other) {
    raw_repr_ |= other.raw_repr_;
    return *this;
}

DispatchKeySet DispatchKeySet::operator&(const DispatchKeySet& other) const {
    DispatchKeySet result = *this;
    result &= other;
    return result;
}

DispatchKeySet& DispatchKeySet::operator&=(const DispatchKeySet& other) {
    raw_repr_ &= other.raw_repr_;
    return *this;
}

DispatchKeySet DispatchKeySet::operator-(const DispatchKeySet& other) const {
    DispatchKeySet result = *this;
    result -= other;
    return result;
}

DispatchKeySet& DispatchKeySet::operator-=(const DispatchKeySet& other) {
    raw_repr_ &= ~other.raw_repr_;
    return *this;
}

bool DispatchKeySet::operator==(const DispatchKeySet& other) const {
    return raw_repr_ == other.raw_repr_;
}

bool DispatchKeySet::operator!=(const DispatchKeySet& other) const {
    return !(*this == other);
}

std::vector<DispatchKey> DispatchKeySet::toVector() const {
    std::vector<DispatchKey> result;
    
    // 收集所有存在的dispatch key
    for (int i = 0; i < static_cast<int>(DispatchKey::NumDispatchKeys); ++i) {
        auto key = static_cast<DispatchKey>(i);
        if (has(key)) {
            result.push_back(key);
        }
    }
    
    // 按优先级排序
    std::sort(result.begin(), result.end(),
              [](DispatchKey a, DispatchKey b) {
                  return dispatchKeyPriority(a) < dispatchKeyPriority(b);
              });
    
    return result;
}

std::string DispatchKeySet::toString() const {
    if (empty()) {
        return "{}";
    }
    
    std::ostringstream oss;
    oss << "{";
    
    auto keys = toVector();
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dispatcher::toString(keys[i]);
    }
    
    oss << "}";
    return oss.str();
}

} // namespace dispatcher 