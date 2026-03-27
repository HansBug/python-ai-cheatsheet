# Python AI 算法岗面试速查手册

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-orange.svg)](#)

> 专为 AI 算法岗面试准备的资料库：覆盖核心算法原理、最小实现、训练推导与现场手撕代码

## 这是什么？

这是一个专注于**AI 算法工程岗面试**的 Python 速查仓库。这里说的“算法”不是刷题里的哈希表、线段树那种题型模板，而是更偏：

- 深度学习基础 / 训练机制
- LLM / Transformer
- CV / Detection / Segmentation / Diffusion / Perception
- 图形学 / 计算几何
- C++ / Python 互操作 / OpenMP
- CUDA / 算子 / 推理优化
- 训练稳定性、并行策略、评测与工程落地

这个仓库的目标不是堆概念，而是整理一套在面试里真正有用的资料组织方式：

- **核心原理讲清楚**：知道模块为什么这样设计，而不是只会背结论
- **最小代码能手写**：关键模块要能在现场写出一个简化版实现
- **训练与推理问题能展开**：不仅会讲结构，还能讲复杂度、显存、稳定性和工程 trade-off
- **面试表达导向**：内容组织会更接近“怎么讲给面试官听”，而不是教材式展开
- **高频主题优先**：先收录 AI 算法岗最常见、最容易被追问的方向

## 适合谁？

### 你应该来看这个库，如果你：

- 已经有 Python 和机器学习基础
- 正在准备 AI 算法岗、LLM 算法岗、CV 算法岗、推荐/强化学习算法岗面试
- 看过论文、做过项目，但高频基础问题答得还不够体系化
- 知道 Transformer / PPO / CNN / CUDA 这些名词，但现场很难讲透或手写
- 想把“会用”升级成“会讲、会推、会改、会写简化实现”

### 你不应该只看这个库，如果你：

- 还没有 Python、PyTorch、线性代数、概率统计的基础
- 还没系统学过深度学习和机器学习
- 当前更需要的是从零入门模型、训练和推理基础

如果你属于上面这类情况，建议先补：

- Python 基础：[Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/)
- 深度学习基础：[Dive into Deep Learning](https://zh.d2l.ai/)
- PyTorch 基础：[PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- Transformer 基础：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## 核心原则

1. **先讲清，再展开，再深入**
   先把核心机制说明白，再补公式、复杂度和工程细节。

2. **最小实现必须可手写**
   注意力、LayerNorm、RoPE、PPO loss 这类高频模块，应该至少能写出简化版。

3. **默认同时覆盖原理与工程**
   每份内容都尽量回答：原理是什么、训练时会遇到什么、线上部署时要注意什么。

4. **大厂面试高频优先于百科全书**
   不追求一次收全所有方向，优先收录最容易被问、最容易被追问的内容。

## 仓库结构

```text
python-ai-cheatsheet/
├── algos/              # 算法方向专题内容
├── tools/              # 本地辅助脚本、实验脚本、可视化脚本
├── README.md           # 项目说明
├── CLAUDE.md           # AI 协作与内容维护规范
├── AGENTS.md           # 指向 CLAUDE.md 的软链接
├── requirements.txt    # 本地实验依赖
├── .gitignore
└── LICENSE
```

## 计划收录的内容

> 当前仓库骨架已搭好，后续会按“AI 面试高频 + 可追问 + 可手写”的标准逐步补齐。

* [ ] 深度学习基础
  * [ ] 反向传播 / 计算图
  * [ ] MLP / 全连接网络
  * [ ] 常见激活函数
  * [ ] 常见损失函数
  * [ ] 常见 Optimizer（SGD / Momentum / Adam / AdamW）
  * [ ] PyTorch `nn.Module` / Autograd 基础
  * [ ] 参数初始化
* [ ] Transformer 基础
  * [x] [Self-Attention / Multi-Head Attention](algos/self_attention/README.md)
  * [x] [Positional Encoding / RoPE](algos/positional_encoding/README.md)
  * [x] [LayerNorm / RMSNorm](algos/normalization/README.md)
  * [x] [Encoder / Decoder 结构与区别](algos/encoder_decoder/README.md)
  * [x] [最小完整 Transformer 实现](algos/transformer_minimal/README.md)
* [ ] LLM 机制与工程
  * [x] [LLM 结构与推理流程](algos/llm/README.md)
  * [ ] KV Cache
  * [ ] MoE
  * [ ] 主流开源 LLM 家族精读
    * [ ] GPT 系列（GPT-1 到 GPT-3.5，作为现代 LLM 起源线）
    * [ ] LLaMA 系列
    * [ ] Qwen 系列
    * [ ] DeepSeek 系列
* [ ] 训练机制与优化
  * [ ] Cross Entropy / Label Smoothing
  * [ ] Adam / AdamW
  * [ ] Learning Rate Scheduler
  * [ ] Gradient Clipping
  * [ ] Mixed Precision
* [ ] 对齐与强化学习
  * [x] [强化学习基础与发展沿革](algos/reinforcement_learning_basics/README.md)
  * [x] [DQN](algos/dqn/README.md)
  * [x] [PPO](algos/ppo/README.md)
  * [ ] DPO
  * [ ] GRPO
  * [ ] Reward Model
  * [ ] Advantage Estimation
* [ ] CV基础
  * [ ] 图像表示 / 颜色空间
  * [ ] 图像滤波 / 边缘检测
  * [ ] 特征点 / 描述子
  * [ ] 单应性 / RANSAC
* [ ] 深度学习CV
  * [x] [CNN 基础模块](algos/cnn_basics/README.md)
  * [x] [ResNet](algos/resnet/README.md)
  * [ ] UNet
  * [ ] YOLO
  * [x] [Vision Transformer](algos/vision_transformer/README.md)
    * [ ] ViT
    * [ ] DeiT
    * [ ] Swin Transformer
    * [ ] MetaFormer / PoolFormer
    * [ ] ConvNeXt
    * [ ] EVA / EVA-02
  * [ ] Detection Head
  * [ ] Diffusion 基础
* [ ] 自动驾驶感知
  * [x] [相机模型 / 投影几何](algos/camera_projection/README.md)
  * [x] [多目标跟踪](algos/multi_object_tracking/README.md)
  * [x] [BEV 感知](algos/bev_perception/README.md)
* [ ] 图形学专题
  * [ ] 二维向量 / 仿射变换
  * [ ] 线段相交 / 点在多边形内
  * [ ] 凸包 / 旋转卡壳
  * [x] [任意多边形面积公式](algos/polygon_area/README.md)
  * [x] [凸多边形相交判定与求交](algos/convex_polygon_intersection/README.md)
  * [ ] 贝塞尔曲线 / 样条
  * [ ] 三维变换基础
  * [ ] 光线与三角形相交
* [ ] 感知算法专题
  * [ ] 卡尔曼滤波 / 扩展卡尔曼滤波
  * [ ] 传感器融合
  * [ ] SLAM 基础
  * [ ] 点云基础
  * [ ] Occupancy / Mapping
* [ ] CUDA 与算子优化
  * [ ] CUDA 内存模型
  * [ ] 常见 kernel 优化思路
  * [ ] Flash Attention 思路
  * [ ] Triton 入门
  * [ ] PyTorch 自定义算子
* [ ] C++专题
  * [x] [C++ 面向对象 / 多态](algos/cpp_oop/README.md)
  * [x] [模板 / 泛型](algos/cpp_templates/README.md)
  * [ ] RAII / 智能指针
  * [ ] 左值右值 / move 语义
  * [x] [OpenMP](algos/openmp/README.md)
  * [x] [pybind11 / Python-C++ 互操作](algos/pybind11/README.md)
  * [ ] C++ 并发基础
* [ ] 分布式训练
  * [ ] Data Parallel
  * [ ] Tensor Parallel
  * [ ] Pipeline Parallel
  * [ ] ZeRO
  * [ ] Checkpointing
* [ ] 推理与部署
  * [ ] Quantization
  * [ ] Speculative Decoding
  * [ ] Serving 架构基础
  * [ ] Batch / Latency / Throughput trade-off
  * [ ] 常见线上问题排查
* [ ] 面试现场手撕专题
  * [ ] 手写 Attention
  * [ ] 手写 LayerNorm
  * [ ] 手写 RoPE
  * [ ] 手写 PPO Loss
  * [ ] 手写 NMS / IoU
* [ ] 面试高频追问
  * [ ] 为什么 AdamW 要解耦 weight decay
  * [ ] 为什么 Pre-LN 更稳
  * [ ] PPO 为什么需要 clip
  * [ ] Flash Attention 为什么省显存
  * [ ] KV Cache 为什么能提速

## 使用方式

1. 先看某个专题的 `README.md`，理解它解决什么问题、为什么这样写。
2. 再看最小实现，确认自己能不看提示写出核心部分。
3. 对着模块训练口头表达：输入输出、公式、复杂度、优缺点、适用场景。
4. 针对训练与推理两个维度，补齐常见追问。
5. 如果是 CUDA / 算子类内容，再额外补内存访问、并行粒度和性能瓶颈分析。

## 依赖说明

本仓库中的代码会按主题选择合适依赖：

- `numpy` 用于数值推导和最小实现
- `torch` 用于模型模块、训练逻辑和算子实验
- `matplotlib` 用于可视化和解释性图示

默认原则是：

- 原理解释尽量独立于具体框架
- 最小实现优先使用 `numpy` / `torch`
- 如果某部分天然依赖 CUDA / Triton / C++ 扩展，会明确标注

## 当前状态

目前已完成仓库基础骨架、文档规范和协作说明，并新增了 [Self-Attention / Multi-Head Attention](algos/self_attention/README.md)、[Positional Encoding / RoPE](algos/positional_encoding/README.md)、[LayerNorm / RMSNorm](algos/normalization/README.md)、[Encoder / Decoder 结构与区别](algos/encoder_decoder/README.md)、[最小完整 Transformer 实现](algos/transformer_minimal/README.md)、[LLM 结构与推理流程](algos/llm/README.md)、[强化学习基础与发展沿革](algos/reinforcement_learning_basics/README.md)、[DQN](algos/dqn/README.md)、[PPO](algos/ppo/README.md)、[CNN 基础模块](algos/cnn_basics/README.md)、[ResNet](algos/resnet/README.md)、[Vision Transformer](algos/vision_transformer/README.md)、[相机模型 / 投影几何](algos/camera_projection/README.md)、[多目标跟踪](algos/multi_object_tracking/README.md)、[BEV 感知](algos/bev_perception/README.md)、[C++ 面向对象 / 多态](algos/cpp_oop/README.md)、[模板 / 泛型](algos/cpp_templates/README.md)、[OpenMP](algos/openmp/README.md)、[pybind11 / Python-C++ 互操作](algos/pybind11/README.md)、[任意多边形面积公式](algos/polygon_area/README.md) 和 [凸多边形相交判定与求交](algos/convex_polygon_intersection/README.md) 二十一篇专题内容。后续会继续补深度学习基础、CV 基础、深度学习 CV、LLM / Transformer、感知算法、图形学和 C++ 相关专题。

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。
