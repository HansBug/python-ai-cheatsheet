# OpenMP 面试攻略

## 这是什么？

这是 C++ 里最常见、也最实用的共享内存并行工具之一。

如果只用一句话来讲：

> OpenMP 的核心作用，是用少量 pragma 把原本串行的 CPU 循环并行化，让多核 CPU 能更容易参与数值计算、预处理和推理加速。

它在 AI 相关代码里特别常见，因为很多 CPU 端工作天然就是：

- 大循环
- 独立元素计算
- reduction
- 批量预处理

## 核心机制

### 1. OpenMP 解决什么问题？

很多 CPU 代码的瓶颈不是算法本身，而是：

- 明明每个元素可以独立处理
- 但代码还是单线程在跑

OpenMP 的思路是：

> 用编译器指令告诉系统“这段循环可以并行”，让多核 CPU 一起做。

### 2. 最常见的写法是什么？

最常见的就是：

```cpp
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    ...
}
```

这表示：

- 开一个并行区域
- 把 `for` 循环迭代分给多个线程执行

### 3. reduction 是干什么的？

如果多个线程都要对同一个量累计，比如：

- 求和
- 求最大值
- 统计计数

就不能直接共享写，否则会有 data race。

这时应该写：

```cpp
#pragma omp parallel for reduction(+:sum)
```

意思是：

- 每个线程先维护自己的局部 `sum`
- 最后再按 `+` 规则合并

### 4. `shared` / `private` 是什么？

OpenMP 里变量作用域很重要。

- `shared`：所有线程共享同一个变量
- `private`：每个线程有自己的副本

如果变量既被多个线程写，又没有正确 reduction 或 private 化，就会出现 race condition。

### 5. AI 工程里 OpenMP 常用在哪？

几个高频场景：

- CPU 端数据预处理
- 图像 resize / normalize / decode 后处理
- 简单数值 kernel
- 检索、打分、统计类循环
- Python 扩展模块里的 CPU 加速逻辑

它特别适合：

> 数据规模大、循环规则、每轮迭代之间依赖少的 CPU 端工作。

## 面试高频问题

### 1. OpenMP 和 CUDA 的区别是什么？

OpenMP 主要是 CPU 共享内存并行；CUDA 主要是 GPU 并行计算。

### 2. OpenMP 最常见的 bug 是什么？

数据竞争。多个线程同时写共享变量但没有同步或 reduction。

### 3. 什么时候 `parallel for` 效果不好？

- 迭代工作量太小
- 线程创建 / 调度开销反而更大
- 循环体里有强依赖
- 内存带宽而不是计算成为瓶颈

### 4. `schedule` 是干什么的？

它控制迭代怎么分配给线程，比如静态分配还是动态分配。

## 最小实现

下面这个最小例子做了两件事：

- 并行计算平方和
- 并行做向量缩放

```cpp
float squared_sum(const std::vector<float>& values) {
    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        sum += values[i] * values[i];
    }
    return sum;
}
```

这里最关键的是：

- `parallel for`：把循环并行化
- `reduction(+:sum)`：解决并行累加的竞争问题

再看向量缩放：

```cpp
void scale_inplace(std::vector<float>& values, float scale) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        values[i] *= scale;
    }
}
```

这段能直接并行，是因为：

- 每个线程只写不同位置
- 元素之间没有依赖

完整代码见：[minimal.cpp](minimal.cpp)

## 工程关注点

### 1. 编译时要开 OpenMP 开关

GCC / Clang 常见是：

```bash
c++ -std=c++17 -fopenmp minimal.cpp -o minimal
```

### 2. 并行不是越多线程越好

线程太多时，调度和带宽开销可能反而更差。

### 3. Python 扩展里要小心线程模型

如果 OpenMP 和 Python / NumPy / PyTorch 自己的线程池叠加，容易过度并行。

## 常见坑点

### 1. 忘了 reduction

并行求和最容易写错。

### 2. 把有依赖的循环直接并行化

这会直接产生错误结果。

### 3. 忽略内存带宽瓶颈

很多 CPU 循环不是算得慢，而是读写内存太慢。

## 面试时怎么讲

如果面试官问 OpenMP，可以按这个顺序讲：

1. OpenMP 是 C++ 里常见的共享内存并行工具
2. 最常用的是 `#pragma omp parallel for`
3. 对于求和这类操作，要用 `reduction`
4. 它适合 CPU 上独立迭代的大循环，比如图像预处理和简单数值计算
5. 最大风险是数据竞争和线程开销不划算

一个简洁版本可以直接讲：

> OpenMP 通过 pragma 让 CPU 循环更容易并行化，最常见写法是 `parallel for`。如果多个线程要共同累计结果，就需要用 `reduction` 避免数据竞争。它很适合 AI 工程里 CPU 端的大规模独立循环，比如预处理、后处理和简单数值 kernel，但要注意线程开销和内存带宽瓶颈。

## 延伸阅读

- pybind11 / Python-C++ 互操作：[pybind11 / Python-C++ 互操作](../pybind11/README.md)
- 配套代码：[minimal.cpp](minimal.cpp)
