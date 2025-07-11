#include "IValue.h"
#include "TensorImpl.h"
#include <stdexcept>
#include <sstream>

namespace dispatcher {

// IValue构造函数实现
IValue::IValue(std::shared_ptr<TensorImpl> tensor) : tag_(Tag::Tensor) {
    payload_.tensor = new std::shared_ptr<TensorImpl>(std::move(tensor));
}

IValue::IValue(double value) : tag_(Tag::Double) {
    payload_.as_double = value;
}

IValue::IValue(int64_t value) : tag_(Tag::Int) {
    payload_.as_int = value;
}

IValue::IValue(bool value) : tag_(Tag::Bool) {
    payload_.as_bool = value;
}

IValue::IValue(std::string value) : tag_(Tag::String) {
    payload_.as_string = new std::string(std::move(value));
}

IValue::IValue(const char* value) : tag_(Tag::String) {
    payload_.as_string = new std::string(value);
}

IValue::IValue(std::vector<int64_t> value) : tag_(Tag::IntList) {
    payload_.as_int_list = new std::vector<int64_t>(std::move(value));
}

IValue::IValue(std::vector<double> value) : tag_(Tag::DoubleList) {
    payload_.as_double_list = new std::vector<double>(std::move(value));
}

IValue::IValue(std::vector<std::shared_ptr<TensorImpl>> value) : tag_(Tag::TensorList) {
    payload_.as_tensor_list = new std::vector<std::shared_ptr<TensorImpl>>(std::move(value));
}

// 拷贝构造函数
IValue::IValue(const IValue& other) : tag_(other.tag_) {
    switch (tag_) {
        case Tag::None:
            break;
        case Tag::Tensor:
            payload_.tensor = new std::shared_ptr<TensorImpl>(*other.payload_.tensor);
            break;
        case Tag::Double:
            payload_.as_double = other.payload_.as_double;
            break;
        case Tag::Int:
            payload_.as_int = other.payload_.as_int;
            break;
        case Tag::Bool:
            payload_.as_bool = other.payload_.as_bool;
            break;
        case Tag::String:
            payload_.as_string = new std::string(*other.payload_.as_string);
            break;
        case Tag::IntList:
            payload_.as_int_list = new std::vector<int64_t>(*other.payload_.as_int_list);
            break;
        case Tag::DoubleList:
            payload_.as_double_list = new std::vector<double>(*other.payload_.as_double_list);
            break;
        case Tag::TensorList:
            payload_.as_tensor_list = new std::vector<std::shared_ptr<TensorImpl>>(*other.payload_.as_tensor_list);
            break;
    }
}

// 拷贝赋值运算符
IValue& IValue::operator=(const IValue& other) {
    if (this != &other) {
        destroy();  // 清理当前数据
        tag_ = other.tag_;
        
        // 拷贝新数据（与拷贝构造函数相同的逻辑）
        switch (tag_) {
            case Tag::None:
                break;
            case Tag::Tensor:
                payload_.tensor = new std::shared_ptr<TensorImpl>(*other.payload_.tensor);
                break;
            case Tag::Double:
                payload_.as_double = other.payload_.as_double;
                break;
            case Tag::Int:
                payload_.as_int = other.payload_.as_int;
                break;
            case Tag::Bool:
                payload_.as_bool = other.payload_.as_bool;
                break;
            case Tag::String:
                payload_.as_string = new std::string(*other.payload_.as_string);
                break;
            case Tag::IntList:
                payload_.as_int_list = new std::vector<int64_t>(*other.payload_.as_int_list);
                break;
            case Tag::DoubleList:
                payload_.as_double_list = new std::vector<double>(*other.payload_.as_double_list);
                break;
            case Tag::TensorList:
                payload_.as_tensor_list = new std::vector<std::shared_ptr<TensorImpl>>(*other.payload_.as_tensor_list);
                break;
        }
    }
    return *this;
}

// 移动构造函数
IValue::IValue(IValue&& other) noexcept : tag_(other.tag_), payload_(other.payload_) {
    other.tag_ = Tag::None;  // 重置other为None状态，避免析构时重复释放
}

// 移动赋值运算符
IValue& IValue::operator=(IValue&& other) noexcept {
    if (this != &other) {
        destroy();  // 清理当前数据
        tag_ = other.tag_;
        payload_ = other.payload_;
        other.tag_ = Tag::None;  // 重置other
    }
    return *this;
}

// 析构函数
IValue::~IValue() {
    destroy();
}

// 类型转换函数实现
std::shared_ptr<TensorImpl> IValue::toTensor() const {
    if (!isTensor()) {
        throw std::runtime_error("IValue is not a Tensor");
    }
    return *payload_.tensor;
}

double IValue::toDouble() const {
    if (!isDouble()) {
        throw std::runtime_error("IValue is not a Double");
    }
    return payload_.as_double;
}

int64_t IValue::toInt() const {
    if (!isInt()) {
        throw std::runtime_error("IValue is not an Int");
    }
    return payload_.as_int;
}

bool IValue::toBool() const {
    if (!isBool()) {
        throw std::runtime_error("IValue is not a Bool");
    }
    return payload_.as_bool;
}

std::string IValue::toString() const {
    if (!isString()) {
        throw std::runtime_error("IValue is not a String");
    }
    return *payload_.as_string;
}

std::vector<int64_t> IValue::toIntList() const {
    if (!isIntList()) {
        throw std::runtime_error("IValue is not an IntList");
    }
    return *payload_.as_int_list;
}

std::vector<double> IValue::toDoubleList() const {
    if (!isDoubleList()) {
        throw std::runtime_error("IValue is not a DoubleList");
    }
    return *payload_.as_double_list;
}

std::vector<std::shared_ptr<TensorImpl>> IValue::toTensorList() const {
    if (!isTensorList()) {
        throw std::runtime_error("IValue is not a TensorList");
    }
    return *payload_.as_tensor_list;
}

// 调试字符串表示
std::string IValue::debugString() const {
    switch (tag_) {
        case Tag::None:
            return "None";
        case Tag::Tensor: {
            auto tensor = toTensor();
            return "Tensor(" + tensor->debugString() + ")";
        }
        case Tag::Double:
            return "Double(" + std::to_string(toDouble()) + ")";
        case Tag::Int:
            return "Int(" + std::to_string(toInt()) + ")";
        case Tag::Bool:
            return std::string("Bool(") + (toBool() ? "true" : "false") + ")";
        case Tag::String:
            return "String(\"" + toString() + "\")";
        case Tag::IntList: {
            std::ostringstream oss;
            oss << "IntList([";
            auto list = toIntList();
            for (size_t i = 0; i < list.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << list[i];
            }
            oss << "])";
            return oss.str();
        }
        case Tag::DoubleList: {
            std::ostringstream oss;
            oss << "DoubleList([";
            auto list = toDoubleList();
            for (size_t i = 0; i < list.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << list[i];
            }
            oss << "])";
            return oss.str();
        }
        case Tag::TensorList: {
            std::ostringstream oss;
            oss << "TensorList([";
            auto list = toTensorList();
            for (size_t i = 0; i < list.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << list[i]->debugString();
            }
            oss << "])";
            return oss.str();
        }
        default:
            return "Unknown";
    }
}

// 内部辅助函数 - 清理payload数据
void IValue::destroy() {
    switch (tag_) {
        case Tag::Tensor:
            delete payload_.tensor;
            break;
        case Tag::String:
            delete payload_.as_string;
            break;
        case Tag::IntList:
            delete payload_.as_int_list;
            break;
        case Tag::DoubleList:
            delete payload_.as_double_list;
            break;
        case Tag::TensorList:
            delete payload_.as_tensor_list;
            break;
        default:
            // 基本类型不需要释放内存
            break;
    }
}

} // namespace dispatcher 