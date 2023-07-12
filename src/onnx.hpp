#pragma once

#include "assert.hpp"
#include "environment.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4554)
#endif
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <array>
#include <numeric>
#include <span>

namespace dlimgedit {

template <typename Scalar, int Rank> using Tensor = Eigen::Tensor<Scalar, Rank, Eigen::RowMajor>;
template <typename Scalar, int Rank> using TensorMap = Eigen::TensorMap<Tensor<Scalar, Rank>>;
template <typename Scalar, int Size>
using TensorArray = Eigen::TensorFixedSize<Scalar, Eigen::Sizes<Size>, Eigen::RowMajor>;

class Shape {
  public:
    Shape() = default;
    explicit Shape(int64_t a, int64_t b, int64_t c, int64_t d) : a_{a, b, c, d}, rank_(4) {}
    explicit Shape(int64_t a, int64_t b, int64_t c) : a_{a, b, c}, rank_(3) {}
    explicit Shape(int64_t a, int64_t b) : a_{a, b}, rank_(2) {}
    explicit Shape(int64_t a) : a_{a}, rank_(1) {}
    explicit Shape(Extent e) : a_{e.height, e.width}, rank_(2) {}
    explicit Shape(Extent e, Channels c)
        : a_{e.height, e.width, static_cast<int64_t>(c)}, rank_(3) {}
    Shape(std::span<int64_t const> a) : rank_(a.size()) {
        ASSERT(rank_ <= 4);
        std::copy(a.begin(), a.end(), a_.begin());
    }
    Shape(std::vector<int64_t> const& a) : Shape(std::span<int64_t const>(a)) {}

    template <typename Sizes> static Shape from_eigen(Sizes const& sizes) {
        ASSERT(Sizes::count <= 4);
        Shape shape;
        shape.rank_ = Sizes::count;
        for (int i = 0; i < shape.rank_; ++i) {
            shape.a_[i] = sizes[i];
        }
        return shape;
    }

    int64_t const* data() const { return a_.data(); }
    int64_t operator[](size_t i) const { return a_[i]; }
    int64_t rank() const { return rank_; }
    int64_t element_count() const {
        return std::accumulate(a_.begin(), a_.begin() + rank_, int64_t(1), std::multiplies<>());
    }

    operator std::array<Eigen::DenseIndex, 2>() const {
        ASSERT(rank_ == 2);
        return std::array<Eigen::DenseIndex, 2>{a_[0], a_[1]};
    }
    operator std::array<Eigen::DenseIndex, 3>() const {
        ASSERT(rank_ == 3);
        return std::array<Eigen::DenseIndex, 3>{a_[0], a_[1], a_[2]};
    }
    operator std::array<Eigen::DenseIndex, 4>() const {
        ASSERT(rank_ == 4);
        return a_;
    }

  private:
    std::array<int64_t, 4> a_{1, 1, 1, 1};
    int64_t rank_ = 1;
};

inline int64_t tensor_size(Shape shape, int64_t element_size) {
    return shape.element_count() * element_size;
}

template <typename Tensor> Ort::Value create_input(EnvironmentImpl& env, Tensor const& tensor) {
    // Ort::Value doesn't support const even for input tensors
    using T = typename Tensor::Scalar;
    T* mut = const_cast<T*>(tensor.data());
    auto shape = Shape::from_eigen(tensor.dimensions());
    return Ort::Value::CreateTensor<T>(env.memory_info, mut, tensor.size(), shape.data(),
                                       shape.rank());
}

template <typename T> Ort::Value create_output(EnvironmentImpl& env, T& tensor) {
    static_assert(!std::is_const_v<T>);
    return create_input(env, tensor);
}

template <typename T, int64_t Rank> TensorMap<T, Rank> as_tensor(Ort::Value const& value) {
    auto shape = Shape(value.GetTensorTypeAndShapeInfo().GetShape());
    ASSERT(shape.rank() == Rank);
    return TensorMap<T, Rank>(value.GetTensorData<T>(), shape);
}

template <typename T, int64_t Rank> TensorMap<T, Rank> as_tensor(Ort::Value& value) {
    auto shape = Shape(value.GetTensorTypeAndShapeInfo().GetShape());
    ASSERT(shape.rank() == Rank);
    return TensorMap<T, Rank>(value.GetTensorMutableData<T>(), shape);
}

TensorMap<uint8_t, 3> as_tensor(Image&);
TensorMap<uint8_t const, 3> as_tensor(ImageView);

} // namespace dlimgedit
