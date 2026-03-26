#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("size mismatch");
    }

    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

class AffineOp {
public:
    AffineOp(float scale, float bias) : scale_(scale), bias_(bias) {}

    std::vector<float> apply(const std::vector<float>& values) const {
        std::vector<float> result = values;
        for (float& value : result) {
            value = value * scale_ + bias_;
        }
        return result;
    }

private:
    float scale_;
    float bias_;
};

PYBIND11_MODULE(minimal_cpp_ext, m) {
    m.doc() = "minimal pybind11 example";

    m.def("dot_product", &dot_product, py::arg("a"), py::arg("b"));

    py::class_<AffineOp>(m, "AffineOp")
        .def(py::init<float, float>())
        .def("apply", &AffineOp::apply);
}
