#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class TensorOp {
public:
    virtual ~TensorOp() = default;
    virtual void apply(std::vector<float>& values) const = 0;
    virtual std::string name() const = 0;
};

class ScaleOp : public TensorOp {
public:
    explicit ScaleOp(float scale) : scale_(scale) {}

    void apply(std::vector<float>& values) const override {
        for (float& value : values) {
            value *= scale_;
        }
    }

    std::string name() const override {
        return "ScaleOp";
    }

private:
    float scale_;
};

class BiasOp : public TensorOp {
public:
    explicit BiasOp(float bias) : bias_(bias) {}

    void apply(std::vector<float>& values) const override {
        for (float& value : values) {
            value += bias_;
        }
    }

    std::string name() const override {
        return "BiasOp";
    }

private:
    float bias_;
};

int main() {
    std::vector<float> values{1.0f, 2.0f, 3.0f};
    std::vector<std::unique_ptr<TensorOp>> pipeline;
    pipeline.emplace_back(std::make_unique<ScaleOp>(2.0f));
    pipeline.emplace_back(std::make_unique<BiasOp>(1.5f));

    for (const auto& op : pipeline) {
        std::cout << "running " << op->name() << '\n';
        op->apply(values);
    }

    std::cout << "result:";
    for (float value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
    return 0;
}
