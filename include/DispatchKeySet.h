#pragma once

#include "DispatchKey.h"
#include <bitset>
#include <vector>

namespace dispatcher {

// DispatchKeySet - 管理dispatch key的集合
// 使用bitset来高效存储和操作dispatch key集合
class DispatchKeySet {
public:
    // 构造函数 - 创建空的dispatch key集合
    DispatchKeySet() = default;
    
    // 构造函数 - 从单个dispatch key创建集合
    explicit DispatchKeySet(DispatchKey key);
    
    // 构造函数 - 从多个dispatch key创建集合
    DispatchKeySet(std::initializer_list<DispatchKey> keys);
    
    // 添加dispatch key到集合中
    DispatchKeySet& add(DispatchKey key);
    
    // 从集合中移除dispatch key
    DispatchKeySet& remove(DispatchKey key);
    
    // 检查集合中是否包含指定的dispatch key
    bool has(DispatchKey key) const;
    
    // 检查集合是否为空
    bool empty() const;
    
    // 清空集合
    void clear();
    
    // 获取最高优先级的dispatch key
    // 按照dispatch key的优先级顺序返回第一个存在的key
    DispatchKey highestPriorityKey() const;
    
    // 集合运算 - 并集
    DispatchKeySet operator|(const DispatchKeySet& other) const;
    DispatchKeySet& operator|=(const DispatchKeySet& other);
    
    // 集合运算 - 交集
    DispatchKeySet operator&(const DispatchKeySet& other) const;
    DispatchKeySet& operator&=(const DispatchKeySet& other);
    
    // 集合运算 - 差集
    DispatchKeySet operator-(const DispatchKeySet& other) const;
    DispatchKeySet& operator-=(const DispatchKeySet& other);
    
    // 比较运算符
    bool operator==(const DispatchKeySet& other) const;
    bool operator!=(const DispatchKeySet& other) const;
    
    // 转换为vector，按优先级排序
    std::vector<DispatchKey> toVector() const;
    
    // 调试用字符串表示
    std::string toString() const;

private:
    // 使用bitset存储dispatch key集合
    std::bitset<static_cast<size_t>(DispatchKey::NumDispatchKeys)> raw_repr_;
    
    // 内部辅助函数 - 获取dispatch key的索引
    static size_t keyIndex(DispatchKey key) {
        return static_cast<size_t>(key);
    }
};

} // namespace dispatcher 