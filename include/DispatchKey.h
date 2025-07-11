#pragma once

#include <cstdint>
#include <string>

namespace dispatcher {

// DispatchKey枚举 - 对应不同的dispatch维度
enum class DispatchKey : uint8_t {
    // Backend keys - 后端特定的dispatch key
    CPU = 0,
    CUDA = 1,
    
    // Functionality keys - 功能性的dispatch key
    Autograd = 2,
    Tracing = 3,
    Profiling = 4,
    
    // Special keys - 特殊的dispatch key
    Undefined = 5,
    CatchAll = 6,
    
    // 总数量，用于数组大小
    NumDispatchKeys = 7
};

// 将DispatchKey转换为字符串，用于调试
const char* toString(DispatchKey key);

// 获取DispatchKey的优先级，数字越小优先级越高
uint8_t dispatchKeyPriority(DispatchKey key);

// 检查是否是Backend dispatch key
bool isBackendKey(DispatchKey key);

// 检查是否是Functionality dispatch key  
bool isFunctionalityKey(DispatchKey key);

} // namespace dispatcher 