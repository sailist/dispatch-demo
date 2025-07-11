#include "OperatorHandle.h"
#include "TensorImpl.h"
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace dispatcher {

// KernelFunction实现
KernelFunction::KernelFunction(BoxedKernelFunction boxed_fn)
    : boxed_fn_(std::move(boxed_fn)) {
}

IValueList KernelFunction::callBoxed(const IValueList& args) const {
    if (!isValid()) {
        throw std::runtime_error("Attempting to call invalid KernelFunction");
    }
    return boxed_fn_(args);
}

// OperatorHandle实现
OperatorHandle::OperatorHandle(std::string name) : name_(std::move(name)) {
}

void OperatorHandle::setKernel(DispatchKey key, KernelFunction kernel) {
    // 将内核函数注册到指定的dispatch key
    dispatch_table_[key] = std::move(kernel);
}

void OperatorHandle::removeKernel(DispatchKey key) {
    // 从dispatch table中移除指定的内核
    dispatch_table_.erase(key);
}

bool OperatorHandle::hasKernel(DispatchKey key) const {
    // 检查是否存在指定dispatch key的内核
    return dispatch_table_.find(key) != dispatch_table_.end();
}

const KernelFunction* OperatorHandle::findKernel(const DispatchKeySet& ks) const {
    // 这是dispatch的核心逻辑：按优先级顺序查找第一个可用的内核
    
    // 获取按优先级排序的dispatch key列表
    auto keys = ks.toVector();
    
    // 按优先级顺序查找第一个有对应内核的dispatch key
    for (auto key : keys) {
        auto it = dispatch_table_.find(key);
        if (it != dispatch_table_.end()) {
            return &it->second;
        }
    }
    
    // 如果没有找到匹配的内核，尝试CatchAll内核作为fallback
    auto catch_all_it = dispatch_table_.find(DispatchKey::CatchAll);
    if (catch_all_it != dispatch_table_.end()) {
        return &catch_all_it->second;
    }
    
    // 没有找到任何匹配的内核
    return nullptr;
}

IValueList OperatorHandle::call(const DispatchKeySet& ks, const IValueList& args) const {
    // 查找匹配的内核函数
    const KernelFunction* kernel = findKernel(ks);
    
    if (!kernel) {
        throw std::runtime_error("No kernel found for operator '" + name_ + 
                               "' with dispatch key set " + ks.toString());
    }
    
    // 调用找到的内核函数
    return kernel->callBoxed(args);
}

IValueList OperatorHandle::call(const IValueList& args) const {
    // 从参数自动计算dispatch key set
    DispatchKeySet ks = computeDispatchKeySet(args);
    return call(ks, args);
}

std::vector<DispatchKey> OperatorHandle::getRegisteredKeys() const {
    std::vector<DispatchKey> keys;
    
    // 收集所有已注册的dispatch key
    for (const auto& entry : dispatch_table_) {
        keys.push_back(entry.first);
    }
    
    // 按优先级排序
    std::sort(keys.begin(), keys.end(),
              [](DispatchKey a, DispatchKey b) {
                  return dispatchKeyPriority(a) < dispatchKeyPriority(b);
              });
    
    return keys;
}

std::string OperatorHandle::debugString() const {
    std::ostringstream oss;
    oss << "OperatorHandle(" << name_ << ") {\n";
    
    auto keys = getRegisteredKeys();
    for (auto key : keys) {
        oss << "  " << toString(key) << ": registered\n";
    }
    
    oss << "}";
    return oss.str();
}

DispatchKeySet OperatorHandle::computeDispatchKeySet(const IValueList& args) const {
    std::vector<std::shared_ptr<TensorImpl>> tensors;
    
    // 从参数中提取所有tensor
    for (const auto& arg : args) {
        if (arg.isTensor()) {
            tensors.push_back(arg.toTensor());
        } else if (arg.isTensorList()) {
            // 如果参数是tensor列表，展开所有tensor
            auto tensor_list = arg.toTensorList();
            tensors.insert(tensors.end(), tensor_list.begin(), tensor_list.end());
        }
    }
    
    // 使用utility函数计算合并的dispatch key set
    return dispatcher::computeDispatchKeySet(tensors);
}

} // namespace dispatcher 