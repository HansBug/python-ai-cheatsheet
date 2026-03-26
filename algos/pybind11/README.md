# pybind11 / Python-C++ 互操作 面试攻略

## 这是什么？

这是把 C++ 能力暴露给 Python 的一条非常常见的工程路径。

如果只用一句话来讲：

> pybind11 的核心作用，是用很薄的一层绑定代码，把 C++ 函数、类和数据结构暴露成 Python 可直接调用的模块。

它在 AI / Python 工程里非常常见，因为很多时候你会遇到这种需求：

- Python 负责训练 / 调度
- 某段 CPU 逻辑想用 C++ 提速
- 想把已有 C++ 库接进 Python
- 想给 Python 提供更低层、更高性能的实现

## 核心机制

### 1. pybind11 解决什么问题？

Python 开发快，但有些底层逻辑更适合用 C++ 写。

pybind11 的作用就是：

> 让你不用手写 CPython C API，也能把 C++ 代码包装成 Python 扩展模块。

### 2. 最基础的绑定长什么样？

一个 pybind11 模块通常长这样：

```cpp
namespace py = pybind11;

PYBIND11_MODULE(minimal_cpp_ext, m) {
    m.def("dot_product", &dot_product, py::arg("a"), py::arg("b"));
}
```

这里几件事要讲清：

- `namespace py = pybind11;`：给命名空间起别名，写起来更短
- `PYBIND11_MODULE(...)`：定义 Python 可导入的模块入口
- `m.def(...)`：把一个 C++ 函数注册成 Python 函数

### 3. 类怎么暴露给 Python？

最常见的是：

```cpp
py::class_<AffineOp>(m, "AffineOp")
    .def(py::init<float, float>())
    .def("apply", &AffineOp::apply);
```

意思是：

- Python 里会得到一个 `AffineOp` 类
- 构造函数映射到 `py::init<...>()`
- 成员函数通过 `.def(...)` 暴露

### 4. AI / Python 场景里通常怎么用？

几个典型场景：

- CPU 预处理 / 后处理逻辑下沉到 C++
- 检索、采样、排序、解码等性能敏感部分下沉
- 把已有 C++ 推理库包一层 Python 接口

要点是：

> pybind11 最适合当“Python 调用 C++”的桥，而不是让你在绑定层做复杂业务逻辑。

## 面试高频问题

### 1. pybind11 和 ctypes / cffi 有什么区别？

pybind11 更偏现代 C++ 风格，直接绑定 C++ 类和函数更自然；ctypes / cffi 更偏 C 风格接口。

### 2. 为什么 AI 工程里常用 pybind11？

因为很多项目主流程在 Python，但部分性能敏感逻辑适合放在 C++。

### 3. pybind11 能直接绑定复杂 C++ 类吗？

可以，但越复杂的所有权、生命周期和异常传播，绑定层越要小心。

### 4. pybind11 一定比纯 Python 快吗？

只有真正把重计算放进 C++ 才有意义。绑定层本身不是魔法。

## 最小实现

下面这个最小例子暴露了一个函数和一个类：

```cpp
namespace py = pybind11;

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    ...
}

class AffineOp {
public:
    AffineOp(float scale, float bias) : scale_(scale), bias_(bias) {}
    std::vector<float> apply(const std::vector<float>& values) const { ... }
};

PYBIND11_MODULE(minimal_cpp_ext, m) {
    m.def("dot_product", &dot_product, py::arg("a"), py::arg("b"));

    py::class_<AffineOp>(m, "AffineOp")
        .def(py::init<float, float>())
        .def("apply", &AffineOp::apply);
}
```

这段最值得你讲的是：

- `dot_product`：一个最小函数绑定
- `AffineOp`：一个最小类绑定
- `std::vector<float>`：pybind11 会帮助做常见容器转换
- 模块入口里只做注册，不写复杂逻辑

完整代码见：[minimal.cpp](minimal.cpp)

## 工程关注点

### 1. 环境前提要明确

这类代码通常依赖：

- Python 开发头文件
- `pybind11` 头文件
- 正确的编译参数

而且 `pybind11` 版本最好和 Python 版本匹配。否则代码本身没问题，编译也可能直接卡在头文件兼容性上。

### 2. 绑定层要尽量薄

业务逻辑尽量留在纯 C++ 代码里，绑定层只负责接口转换。

### 3. 生命周期和异常传播要小心

这通常是 C++ / Python 互操作里最容易出问题的部分。

## 常见坑点

### 1. 把绑定层写得太重

这样维护成本会很高。

### 2. 忘记模块名要和 Python 导入名对应

`PYBIND11_MODULE(name, m)` 里的 `name` 会直接影响导入。

### 3. 误以为“包一层就一定快”

真正耗时的逻辑没下沉到 C++，那就不会有本质收益。

## 面试时怎么讲

如果面试官问 pybind11，可以按这个顺序讲：

1. pybind11 是 C++ 到 Python 的绑定工具
2. 它能比较自然地暴露 C++ 函数和类
3. 在 AI 工程里，常用于把 CPU 端性能敏感逻辑下沉到 C++
4. 绑定层最好保持很薄，避免把复杂业务逻辑写进接口层
5. 真正难点通常在编译环境、生命周期和数据转换

一个简洁版本可以直接讲：

> pybind11 的作用是把 C++ 函数和类包装成 Python 可直接导入的扩展模块。它在 AI 工程里很常见，因为很多系统主流程在 Python，但某些 CPU 端性能敏感逻辑更适合用 C++ 实现。实际工程里最好把绑定层写薄，只做接口注册和数据转换，真正的逻辑仍然放在纯 C++ 代码里。

## 延伸阅读

- OpenMP：[OpenMP](../openmp/README.md)
- C++ 面向对象 / 多态：[C++ 面向对象 / 多态](../cpp_oop/README.md)
- 配套代码：[minimal.cpp](minimal.cpp)
