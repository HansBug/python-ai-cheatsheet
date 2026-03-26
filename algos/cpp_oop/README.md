# C++ 面向对象 / 多态 面试攻略

## 这是什么？

这是 AI 工程和 C++ 面试里最常见的一类基础题。

这里重点不是背定义，而是要能说清：

- 类、继承、封装到底解决什么问题
- 动态多态为什么需要 `virtual`
- 什么时候该用运行时多态，什么时候不该用
- 在 AI / Python 相关代码里，多态通常会被用在什么地方

如果只用一句话来讲：

> C++ 面向对象的核心是用类组织状态和行为，用继承表达接口关系，用虚函数实现运行时多态，让同一套调用代码可以统一处理不同实现。

## 核心机制

### 1. 类在 C++ 里到底解决什么问题？

最基础的类，就是把数据和操作打包在一起。

比如在 AI 代码里，一个“算子”通常不只是一个函数，还会带：

- 参数
- 配置
- 生命周期
- 统一调用接口

这时用类来表达会更自然。

如果先从代码看，一个最小的抽象接口可以长这样：

```cpp
class TensorOp {
public:
    virtual ~TensorOp() = default;
    virtual void apply(std::vector<float>& values) const = 0;
    virtual std::string name() const = 0;
};
```

这几行代码本身就对应了面向对象里最重要的几个概念：

- `class TensorOp`：把“算子”抽象成一个对象，而不是散落的函数
- `virtual ~TensorOp() = default;`：声明一个虚析构函数，并让编译器直接生成默认析构实现
- `apply(...)`：定义统一行为接口，表示“这个算子能作用在一组数据上”
- `name()`：给外部系统一个统一查询元信息的入口
- `= 0`：把它声明成纯虚函数，说明这里只定义接口，不给默认实现

所以这里的类不是为了“语法上像 C++”，而是为了把“统一接口 + 不同实现”这件事表达清楚。

### 2. 什么是继承？

继承的核心是：

> 用一个更抽象的基类定义公共接口，让具体子类去给出不同实现。

在工程里，这类场景非常常见：

- 不同后端的算子实现
- 不同数据增强策略
- 不同调度器 / 策略类

如果继续对着代码看，继承就是让具体子类接到这个抽象接口下面：

```cpp
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
```

这里最该结合代码点出来的是：

- `public TensorOp`：说明 `ScaleOp` 是一个 `TensorOp`，也就是它遵守这套统一接口
- `explicit ScaleOp(float scale)`：构造函数把这个算子的内部状态 `scale_` 绑定进对象
- `apply(...) const override`：真正给出具体实现，这就是“接口不变、实现可替换”
- `private: float scale_`：把状态封装在对象内部，而不是暴露给外部随便改

所以继承在这里的价值，不是单纯复用代码，而是：

> 让调用方只依赖 `TensorOp` 接口，而不用关心后面到底是 `ScaleOp` 还是别的实现。

### 3. 什么是多态？

最常问的是运行时多态，也就是：

- 基类指针 / 引用指向子类对象
- 调用虚函数时，真正执行的是子类重写版本

这件事依赖 `virtual`。

如果没有 `virtual`，通过基类接口调用时就不会发生动态分派。

真正体现“多态”的，不是子类定义本身，而是这种调用方式：

```cpp
std::vector<std::unique_ptr<TensorOp>> pipeline;
pipeline.emplace_back(std::make_unique<ScaleOp>(2.0f));
pipeline.emplace_back(std::make_unique<BiasOp>(1.5f));

for (const auto& op : pipeline) {
    std::cout << "running " << op->name() << '\n';
    op->apply(values);
}
```

这里必须结合代码讲：

- `std::unique_ptr<TensorOp>`：容器里存的是基类指针，不是具体子类类型
- `make_unique<ScaleOp>(...)` / `make_unique<BiasOp>(...)`：实际塞进去的是不同子类对象
- `op->name()` / `op->apply(values)`：调用发生在基类接口上，但真正执行的是各自子类实现

这就是运行时多态最典型的样子：

> 同一段调用代码，不需要写 `if op_type == ...`，也不需要知道真实子类类型，只通过统一接口就能调到正确实现。

### 4. 为什么析构函数常常也要 `virtual`？

这是高频坑点。

如果你通过基类指针删除子类对象，而基类析构函数不是虚的，就可能只执行到基类析构，导致资源释放不完整。

所以只要这个类打算被当成多态基类使用，析构函数通常就应该是 `virtual`。

对应到前面的代码，就是这一行：

```cpp
virtual ~TensorOp() = default;
```

这里要把它拆开讲，不要混着背：

- `virtual`：说明这是虚析构函数。也就是说，如果你通过 `TensorOp*` 或 `std::unique_ptr<TensorOp>` 去销毁一个真实类型是子类的对象，析构会先动态分派到子类析构，再回到基类析构。
- `= default`：不是“没有析构函数”，而是“让编译器帮你生成一个默认析构实现”。这里基类自己没有额外资源要手动释放，所以默认析构就够了。

也就是说，这一行真正的含义是：

> 这个类需要一个虚析构函数来支持多态销毁，但基类本身不需要手写析构逻辑，所以直接让编译器生成默认版本。

如果你想更直观地看运行逻辑，可以看下面这个继承后的例子：

```cpp
class ScaleOp : public TensorOp {
public:
    explicit ScaleOp(float scale) : scale_(scale) {}

    ~ScaleOp() override {
        std::cout << "destroying ScaleOp\n";
    }
    ...
};

std::unique_ptr<TensorOp> op = std::make_unique<ScaleOp>(2.0f);
```

当 `op` 离开作用域时，运行逻辑是：

1. `op` 的静态类型是 `std::unique_ptr<TensorOp>`，但它持有对象的动态类型是 `ScaleOp`
2. 因为基类析构函数是 `virtual`，销毁时会先进入 `ScaleOp::~ScaleOp()`
3. `ScaleOp` 析构结束后，再继续执行 `TensorOp::~TensorOp()`
4. 而 `TensorOp::~TensorOp()` 因为写的是 `= default`，它的函数体由编译器生成，这里等价于“基类没有额外自定义析构逻辑”

所以真正重要的是：

- `virtual` 决定“会不会正确走到子类析构”
- `= default` 决定“基类析构函数本身由谁来实现”

如果把 `virtual` 去掉，那么通过基类指针 / 基类智能指针销毁子类对象时，运行行为就不再安全了，这才是最危险的点。

### 5. AI / Python 相关代码里，多态一般用在哪？

几个常见场景：

- 统一算子接口：`Op` 基类，下面是 CPU / CUDA / Triton 实现
- 统一策略接口：不同采样、调度、搜索策略
- 插件式模块：不同模型头、后处理器、数据预处理器

要点不是“为了面向对象而面向对象”，而是：

> 当调用方只关心接口，不关心具体实现时，多态最有价值。

## 面试高频问题

### 1. `virtual` 到底干了什么？

它让成员函数支持运行时动态分派，也就是通过基类接口调用时，真正执行子类重写版本。

### 2. 运行时多态和模板多态有什么区别？

- 运行时多态：基于继承和虚函数，灵活，但有一点动态分派开销
- 模板多态：编译期展开，通常更快，但接口是在编译期固定的

### 3. 什么情况下基类析构函数必须是虚的？

只要这个类会被当成多态基类，通过基类指针销毁子类对象，就应该是虚析构。

### 4. 多态一定是好的吗？

不是。它适合接口稳定、实现可替换的场景；如果只是单纯追求性能或编译期静态分发，模板往往更合适。

## 最小实现

这篇最适合对着看的代码锚点其实已经在上面的主讲解里给出来了：

- 抽象基类 `TensorOp`
- 具体子类 `ScaleOp`
- 多态调用容器 `std::vector<std::unique_ptr<TensorOp>>`

建议阅读顺序是：

1. 先看 `TensorOp`，理解接口和虚析构
2. 再看 `ScaleOp` / `BiasOp`，理解继承和重写
3. 最后看 `pipeline` 那段调用，理解多态真正发生在哪里

完整代码见：[minimal.cpp](minimal.cpp)

## 工程关注点

### 1. 多态通常和智能指针一起用

现代 C++ 里更常见的是：

- `std::unique_ptr<Base>`
- `std::shared_ptr<Base>`

而不是裸指针。

### 2. 不要滥用继承

如果只是复用一段代码，组合往往比继承更稳。

### 3. 性能敏感路径要小心虚函数

虚函数有一定运行时分派开销。在极热路径里，模板或内联静态分发可能更合适。

## 常见坑点

### 1. 基类析构函数不是虚的

这是最典型的资源释放 bug。

### 2. 忘记 `override`

容易写出“看起来像重写，实际上没有重写”的隐藏 bug。

### 3. 把继承当成代码复用工具滥用

继承更适合“is-a”关系，不是简单复制逻辑。

## 面试时怎么讲

如果面试官问 C++ 多态，可以按这个顺序讲：

1. 类把状态和行为封装起来
2. 基类定义统一接口，子类提供不同实现
3. 运行时多态依赖虚函数，调用时通过基类接口动态分派到子类实现
4. 多态基类析构函数通常也必须是虚的
5. AI 工程里常见于统一算子、策略和插件式模块接口

一个简洁版本可以直接讲：

> C++ 面向对象里，继承和虚函数主要是为了统一接口和实现可替换。基类给出抽象接口，子类重写具体逻辑；通过基类指针或引用调用时，虚函数会在运行时分派到子类实现。只要类会被当成多态基类使用，析构函数通常也应该是虚的。在 AI 工程里，这种模式常用于统一算子、调度策略和插件式模块接口。

## 延伸阅读

- 模板 / 泛型：[模板 / 泛型](../cpp_templates/README.md)
- pybind11 / Python-C++ 互操作：[pybind11 / Python-C++ 互操作](../pybind11/README.md)
- 配套代码：[minimal.cpp](minimal.cpp)
