#include "TensorImpl.h"
#include <numeric>
#include <sstream>
#include <algorithm>

namespace dispatcher {

// TensorImpl实现
TensorImpl::TensorImpl(std::vector<int64_t> sizes, DispatchKey backend_key)
    : sizes_(std::move(sizes)), backend_key_(backend_key) {
}

int64_t TensorImpl::numel() const {
    if (sizes_.empty()) {
        return 0;
    }
    return std::accumulate(sizes_.begin(), sizes_.end(), 1LL, std::multiplies<int64_t>());
}

DispatchKeySet TensorImpl::keySet() const {
    DispatchKeySet result;
    
    // 第一步：添加backend dispatch key
    result.add(backend_key_);
    
    // 第二步：根据tensor属性添加功能性dispatch key
    if (requires_grad_) {
        result.add(DispatchKey::Autograd);
    }
    
    // 第三步：添加全局状态的功能性dispatch key
    auto global_keys = GlobalDispatchState::instance().computeFunctionalityKeys();
    result |= global_keys;
    
    return result;
}

std::string TensorImpl::debugString() const {
    std::ostringstream oss;
    oss << "shape=[";
    for (size_t i = 0; i < sizes_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << sizes_[i];
    }
    oss << "], backend=" << toString(backend_key_);
    if (requires_grad_) {
        oss << ", requires_grad=true";
    }
    return oss.str();
}

std::shared_ptr<TensorImpl> TensorImpl::clone() const {
    auto cloned = std::make_shared<TensorImpl>(sizes_, backend_key_);
    cloned->setRequiresGrad(requires_grad_);
    return cloned;
}

// 工厂函数实现
std::shared_ptr<TensorImpl> make_tensor_cpu(std::vector<int64_t> sizes) {
    return std::make_shared<TensorImpl>(std::move(sizes), DispatchKey::CPU);
}

std::shared_ptr<TensorImpl> make_tensor_cuda(std::vector<int64_t> sizes) {
    return std::make_shared<TensorImpl>(std::move(sizes), DispatchKey::CUDA);
}

// 工具函数 - 计算多个tensor的合并dispatch key set
DispatchKeySet computeDispatchKeySet(const std::vector<std::shared_ptr<TensorImpl>>& tensors) {
    DispatchKeySet combined_set;
    
    // 遍历所有tensor，合并它们的dispatch key set
    for (const auto& tensor : tensors) {
        if (tensor) {  // 检查tensor是否为空
            combined_set |= tensor->keySet();
        }
    }
    
    // 如果没有tensor，返回只包含全局状态的dispatch key set
    if (combined_set.empty()) {
        combined_set = GlobalDispatchState::instance().computeFunctionalityKeys();
    }
    
    return combined_set;
}

// GlobalDispatchState实现
GlobalDispatchState& GlobalDispatchState::instance() {
    static GlobalDispatchState instance;
    return instance;
}

DispatchKeySet GlobalDispatchState::computeFunctionalityKeys() const {
    DispatchKeySet result;
    
    // 根据全局状态添加对应的dispatch key
    if (autograd_enabled_) {
        result.add(DispatchKey::Autograd);
    }
    
    if (tracing_enabled_) {
        result.add(DispatchKey::Tracing);
    }
    
    if (profiling_enabled_) {
        result.add(DispatchKey::Profiling);
    }
    
    return result;
}

} // namespace dispatcher 