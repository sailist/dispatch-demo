#pragma once

#include <memory>
#include <vector>
#include <variant>
#include <string>
#include <type_traits>

namespace dispatcher {

// 前向声明
class TensorImpl;

// IValue - 实现boxing/unboxing机制
// 允许将任意类型的值包装成统一的IValue类型，用于boxed calling convention
class IValue {
public:
    // 支持的基本类型枚举
    enum class Tag {
        None,
        Tensor,
        Double,
        Int,
        Bool,
        String,
        IntList,
        DoubleList,
        TensorList
    };

    // 构造函数 - 默认构造为None类型
    IValue() : tag_(Tag::None) {}
    
    // 构造函数 - 从tensor构造
    explicit IValue(std::shared_ptr<TensorImpl> tensor);
    
    // 构造函数 - 从基本类型构造
    explicit IValue(double value);
    explicit IValue(int64_t value);
    explicit IValue(bool value);
    explicit IValue(std::string value);
    explicit IValue(const char* value);
    
    // 构造函数 - 从列表类型构造
    explicit IValue(std::vector<int64_t> value);
    explicit IValue(std::vector<double> value);
    explicit IValue(std::vector<std::shared_ptr<TensorImpl>> value);
    
    // 拷贝构造和赋值
    IValue(const IValue& other);
    IValue& operator=(const IValue& other);
    
    // 移动构造和赋值
    IValue(IValue&& other) noexcept;
    IValue& operator=(IValue&& other) noexcept;
    
    // 析构函数
    ~IValue();
    
    // 类型检查函数
    bool isNone() const { return tag_ == Tag::None; }
    bool isTensor() const { return tag_ == Tag::Tensor; }
    bool isDouble() const { return tag_ == Tag::Double; }
    bool isInt() const { return tag_ == Tag::Int; }
    bool isBool() const { return tag_ == Tag::Bool; }
    bool isString() const { return tag_ == Tag::String; }
    bool isIntList() const { return tag_ == Tag::IntList; }
    bool isDoubleList() const { return tag_ == Tag::DoubleList; }
    bool isTensorList() const { return tag_ == Tag::TensorList; }
    
    // 类型转换函数 - 这些函数在类型不匹配时会抛出异常
    std::shared_ptr<TensorImpl> toTensor() const;
    double toDouble() const;
    int64_t toInt() const;
    bool toBool() const;
    std::string toString() const;
    std::vector<int64_t> toIntList() const;
    std::vector<double> toDoubleList() const;
    std::vector<std::shared_ptr<TensorImpl>> toTensorList() const;
    
    // 获取tag
    Tag tag() const { return tag_; }
    
    // 调试用字符串表示
    std::string debugString() const;

private:
    Tag tag_;
    
    // 使用union来存储不同类型的数据，节省内存
    union Payload {
        std::shared_ptr<TensorImpl>* tensor;
        double as_double;
        int64_t as_int;
        bool as_bool;
        std::string* as_string;
        std::vector<int64_t>* as_int_list;
        std::vector<double>* as_double_list;
        std::vector<std::shared_ptr<TensorImpl>>* as_tensor_list;
        
        Payload() {}
        ~Payload() {}
    } payload_;
    
    // 内部辅助函数 - 清理payload中的数据
    void destroy();
};

// 模板函数 - 用于自动boxing
template<typename T>
IValue make_ivalue(T&& value) {
    return IValue(std::forward<T>(value));
}

// IValue列表类型，用于函数参数和返回值
using IValueList = std::vector<IValue>;

} // namespace dispatcher 