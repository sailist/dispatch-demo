#pragma once

#include "DispatchKey.h"
#include "DispatchKeySet.h"
#include <vector>
#include <memory>
#include <string>

namespace dispatcher {

// TensorImpl - 简化的tensor实现
// 用于演示dispatcher如何根据tensor的属性进行分发
class TensorImpl {
public:
    // 构造函数 - 创建指定形状和后端的tensor
    TensorImpl(std::vector<int64_t> sizes, DispatchKey backend_key);
    
    // 虚析构函数，支持继承
    virtual ~TensorImpl() = default;
    
    // 获取tensor的形状
    const std::vector<int64_t>& sizes() const { return sizes_; }
    
    // 获取tensor的元素总数
    int64_t numel() const;
    
    // 获取tensor的维度数量
    int64_t dim() const { return static_cast<int64_t>(sizes_.size()); }
    
    // 获取后端dispatch key（CPU、CUDA等）
    DispatchKey backendKey() const { return backend_key_; }
    
    // 计算当前tensor的dispatch key set
    // 这个函数会根据tensor的属性和全局状态计算出完整的dispatch key集合
    DispatchKeySet keySet() const;
    
    // 设置是否需要梯度（用于autograd）
    void setRequiresGrad(bool requires_grad) { requires_grad_ = requires_grad; }
    bool requiresGrad() const { return requires_grad_; }
    
    // 获取tensor的调试信息
    std::string debugString() const;
    
    // 判断tensor是否在指定设备上
    bool is_cpu() const { return backend_key_ == DispatchKey::CPU; }
    bool is_cuda() const { return backend_key_ == DispatchKey::CUDA; }
    
    // 克隆tensor（浅拷贝metadata，深拷贝数据的接口）
    virtual std::shared_ptr<TensorImpl> clone() const;

protected:
    std::vector<int64_t> sizes_;     // tensor的形状
    DispatchKey backend_key_;        // 后端类型（CPU/CUDA等）
    bool requires_grad_ = false;     // 是否需要梯度计算
    
    // 可以添加其他属性如stride、storage等，这里简化处理
};

// 工厂函数 - 创建不同后端的tensor
std::shared_ptr<TensorImpl> make_tensor_cpu(std::vector<int64_t> sizes);
std::shared_ptr<TensorImpl> make_tensor_cuda(std::vector<int64_t> sizes);

// 工具函数 - 根据tensors计算合并的dispatch key set
DispatchKeySet computeDispatchKeySet(const std::vector<std::shared_ptr<TensorImpl>>& tensors);

// 全局状态管理 - 用于功能性dispatch key
class GlobalDispatchState {
public:
    static GlobalDispatchState& instance();
    
    // 设置和获取是否启用autograd
    void setAutogradEnabled(bool enabled) { autograd_enabled_ = enabled; }
    bool isAutogradEnabled() const { return autograd_enabled_; }
    
    // 设置和获取是否启用tracing
    void setTracingEnabled(bool enabled) { tracing_enabled_ = enabled; }
    bool isTracingEnabled() const { return tracing_enabled_; }
    
    // 设置和获取是否启用profiling
    void setProfilingEnabled(bool enabled) { profiling_enabled_ = enabled; }
    bool isProfilingEnabled() const { return profiling_enabled_; }
    
    // 计算当前全局状态对应的功能性dispatch key set
    DispatchKeySet computeFunctionalityKeys() const;

private:
    bool autograd_enabled_ = false;
    bool tracing_enabled_ = false;
    bool profiling_enabled_ = false;
    
    GlobalDispatchState() = default;
};

} // namespace dispatcher 