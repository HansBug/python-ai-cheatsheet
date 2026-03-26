#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("size mismatch");
    }

    T sum = T{};
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <typename T>
class Tensor1D {
public:
    explicit Tensor1D(std::vector<T> values) : values_(std::move(values)) {}

    std::vector<T>& values() { return values_; }
    const std::vector<T>& values() const { return values_; }

private:
    std::vector<T> values_;
};

struct ScaleAndBias {
    float scale;
    float bias;

    template <typename T>
    T operator()(T value) const {
        return static_cast<T>(value * static_cast<T>(scale) + static_cast<T>(bias));
    }
};

template <typename T, typename UnaryOp>
void apply_inplace(Tensor1D<T>& tensor, UnaryOp op) {
    for (T& value : tensor.values()) {
        value = op(value);
    }
}

int main() {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{4.0f, 5.0f, 6.0f};
    std::cout << "dot(float): " << dot_product(a, b) << '\n';

    std::vector<double> c{1.0, 1.5, 2.0};
    std::vector<double> d{2.0, 2.0, 2.0};
    std::cout << "dot(double): " << dot_product(c, d) << '\n';

    Tensor1D<float> tensor({1.0f, 2.0f, 3.0f});
    apply_inplace(tensor, ScaleAndBias{2.0f, 0.5f});

    std::cout << "transformed:";
    for (float value : tensor.values()) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
    return 0;
}
