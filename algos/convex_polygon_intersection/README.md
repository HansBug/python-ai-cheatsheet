# 凸多边形相交判定与求交 面试攻略

## 这是什么？

这是计算几何里一类非常典型的问题：

- 给两个凸多边形
- 先判断它们是否相交
- 如果相交，再求出交集多边形

如果只用一句话来讲：

> 只做相交判定时，凸多边形最经典的是 SAT；如果还要求出交集形状，更直接的做法是把一个凸多边形依次按另一个凸多边形的每条边对应半平面去裁剪。

## 核心机制

### 1. 问题前提要先说清楚

这里讨论的是**凸多边形**。

设两个多边形分别是：

$$ A = (a_0, a_1, \ldots, a_{m-1}), \quad B = (b_0, b_1, \ldots, b_{n-1}) $$

默认前提：

- 顶点按边界顺序给出
- 最好是逆时针顺序
- 多边形是凸的

之所以凸很重要，是因为很多更简单的判定和求交算法，都依赖“凸集 = 若干半平面的交”这个性质。

### 2. 如果只判定相交，最经典的是 SAT

SAT 是 Separating Axis Theorem，分离轴定理。

它的核心结论是：

> 两个凸多边形不相交，当且仅当存在一条轴，使得它们投影到这条轴上的区间不重叠。

对二维凸多边形来说，只需要检查：

- 多边形 `A` 每条边的法向量
- 多边形 `B` 每条边的法向量

这就够了。

如果先对着代码看，判定逻辑是这样的：

```python
def convex_polygons_intersect(poly_a, poly_b, eps=1e-9):
    poly_a = ensure_ccw(poly_a)
    poly_b = ensure_ccw(poly_b)

    for polygon in (poly_a, poly_b):
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            edge = subtract(p2, p1)
            axis = (-edge[1], edge[0])
            if has_separating_axis(axis, poly_a, poly_b, eps):
                return False
    return True
```

这段代码最该结合原理讲：

- `edge = subtract(p2, p1)`：拿到一条边
- `axis = (-edge[1], edge[0])`：把边旋转 `90` 度得到法向量，也就是候选分离轴
- `has_separating_axis(...)`：把两个多边形都投影到这条轴上，看区间是否分开
- 只要有一条轴分开，就能立刻返回 `False`

### 3. 为什么只检查边法向量就够了？

这是 SAT 最关键的证明点。

如果两个凸多边形不相交，那么一定存在一条分离直线，把它们放在两侧。

把这条分离直线平移，直到它刚好贴住其中一个多边形的边界。  
因为多边形是凸的，这时这条支撑线会和某一条边平行。

于是分离轴的方向，就可以等价地看成某条边的法向量方向。

所以对凸多边形来说：

> 真正需要检查的候选轴，不用枚举所有方向，只要检查所有边的法向量就够了。

### 4. 如果要求交集多边形，更直接的是半平面裁剪

只判定相交时，SAT 很自然。  
但如果你不只是要 `True / False`，还要真正求出交集多边形的顶点，那更直接的思路是：

> 把多边形 `A` 依次按多边形 `B` 的每一条边对应的“内部半平面”去裁剪。

因为凸多边形 `B` 本身就可以看成这些内部半平面的交：

$$ B = H_0 \cap H_1 \cap \cdots \cap H_{n-1} $$

于是：

$$ A \cap B = A \cap H_0 \cap H_1 \cap \cdots \cap H_{n-1} $$

这就是 Sutherland-Hodgman 裁剪算法在这里的核心思想。

### 5. 单次裁剪到底在做什么？

设裁剪边是 `C_i -> C_{i+1}`。  
如果 `B` 是逆时针顺序，那么这条边的“内部”就是它左侧的半平面。

判断一个点 `P` 是否在内部，可以直接看叉积符号：

$$ (C_{i+1} - C_i) \times (P - C_i) \ge 0 $$

对应代码就是：

```python
def inside_half_plane(a, b, p, eps=1e-9):
    return cross(subtract(b, a), subtract(p, a)) >= -eps
```

所以每次裁剪的任务就是：

- 输入一个当前多边形
- 看它每条边的两个端点相对裁剪半平面是 inside 还是 outside
- 决定保留哪些点，是否补一个交点

### 6. 为什么会有“四种情况”？

对当前边 `prev -> curr`，只需要看两个端点的 inside / outside 关系：

1. `prev` inside，`curr` inside  
   保留 `curr`

2. `prev` inside，`curr` outside  
   只保留和裁剪边的交点

3. `prev` outside，`curr` inside  
   先保留交点，再保留 `curr`

4. `prev` outside，`curr` outside  
   什么都不保留

对应代码就是：

```python
def clip_with_half_plane(subject, a, b, eps=1e-9):
    output = []
    prev = subject[-1]
    prev_inside = inside_half_plane(a, b, prev, eps)

    for curr in subject:
        curr_inside = inside_half_plane(a, b, curr, eps)

        if prev_inside and curr_inside:
            output.append(curr)
        elif prev_inside and not curr_inside:
            output.append(line_intersection(prev, curr, a, b))
        elif not prev_inside and curr_inside:
            output.append(line_intersection(prev, curr, a, b))
            output.append(curr)

        prev = curr
        prev_inside = curr_inside

    return output
```

这段代码要结合几何直觉一起讲：

- 从 inside 到 outside：说明边穿出了半平面，所以只留下“出界前最后那一点”，也就是交点
- 从 outside 到 inside：说明边穿进了半平面，所以先补交点，再收下 `curr`
- 两端都 inside：整条边都有效
- 两端都 outside：这条边对结果没有贡献

### 7. 为什么连续裁剪后就是最终交集？

这个证明其实很整齐，可以直接用归纳法。

设 `S_k` 表示裁剪完前 `k` 条边之后的结果。

#### 基础情况

一开始：

$$ S_0 = A $$

#### 归纳步骤

假设裁完前 `k` 条边后：

$$ S_k = A \cap H_0 \cap \cdots \cap H_{k-1} $$

再用第 `k` 条边对应的半平面 `H_k` 去裁：

$$ S_{k+1} = S_k \cap H_k $$

所以：

$$ S_{k+1} = A \cap H_0 \cap \cdots \cap H_k $$

裁完所有边后就得到：

$$ S_n = A \cap B $$

这就证明了裁剪算法的正确性。

### 8. 最终求交代码长什么样？

把前面的局部逻辑串起来，完整主线就是：

```python
def convex_polygon_intersection(subject, clip, eps=1e-9):
    if not subject or not clip:
        return []

    output = ensure_ccw(subject)
    clip = ensure_ccw(clip)

    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        output = clip_with_half_plane(output, a, b, eps)
        if not output:
            return []

    return cleanup_polygon(output, eps)
```

它的运行逻辑其实很直白：

1. 先保证两个凸多边形都是逆时针
2. 用 `clip` 的每一条边去裁 `subject`
3. 每裁一次，`output` 都变成“更小但更接近最终交集”的多边形
4. 全部裁完后，留下来的就是交集多边形

## 面试高频问题

### 1. 为什么凸多边形相交判定常用 SAT？

因为凸集不相交时一定存在分离轴，而二维凸多边形只需要检查所有边法向量即可。

### 2. 如果不只判定相交，还要交集多边形怎么办？

最直接的是 Sutherland-Hodgman 半平面裁剪。

### 3. 为什么裁剪时只看“左侧半平面”？

因为这里默认顶点按逆时针给出，对每条边来说，多边形内部就在边的左侧。

### 4. 这个求交算法对非凸多边形也适用吗？

这里给出的版本默认 clip polygon 是凸的。非凸情形要复杂得多，不能直接照搬这套最小实现。

### 5. 为什么只靠 SAT 不够？

SAT 只能告诉你“有没有交集”，但不会直接给你交集多边形的顶点。

## 最小实现

配套代码里同时给了两套能力：

- `convex_polygons_intersect(...)`：只做 SAT 判定
- `convex_polygon_intersection(...)`：用半平面裁剪求交集多边形

最值得手写和讲清的部分是：

```python
def has_separating_axis(axis, poly_a, poly_b, eps=1e-9):
    min_a, max_a = project_polygon(axis, poly_a)
    min_b, max_b = project_polygon(axis, poly_b)
    return max_a < min_b - eps or max_b < min_a - eps
```

和：

```python
def clip_with_half_plane(subject, a, b, eps=1e-9):
    ...
```

前者体现 SAT 的本质，后者体现求交的本质。

完整代码见：[minimal.py](minimal.py)

## 工程关注点

### 1. 顶点顺序最好统一成逆时针

这样“内部在左侧”这个判断才稳定。

### 2. 浮点误差要加 `eps`

特别是在：

- 边几乎平行
- 点几乎落在边上
- 多边形非常扁时

### 3. 退化情况要单独考虑

最小实现主要面向一般位置和非退化面状交集。  
如果只是点接触、边重合、强共线，工程上通常还要补更细的去重和退化处理。

## 常见坑点

### 1. 忘了凸性前提

SAT 和这里的半平面裁剪版本都默认凸多边形。

### 2. 顶点顺序不一致

顺逆时针混了以后，半平面内外判断会反。

### 3. 只会写判定，不会解释为什么边法向量就够了

这是 SAT 最容易被追问的点。

### 4. 只会说“裁剪”两个字，不会解释四种 inside / outside 情况

这会显得对算法只是背名字，没有真正理解。

## 面试时怎么讲

如果面试官问凸多边形相交，可以按这个顺序讲：

1. 只做判定时，最经典的是 SAT
2. 核心结论是：两个凸多边形不相交，当且仅当存在分离轴
3. 对二维凸多边形，只检查所有边法向量就够了
4. 如果还要求交集多边形，更直接的是把一个多边形按另一个多边形每条边对应的内部半平面去裁剪
5. 每次裁剪只要处理边段的四种 inside / outside 情况
6. 连续裁完所有边后，得到的就是最终交集

一个简洁版本可以直接讲：

> 凸多边形只做相交判定时，可以用 SAT：如果存在一条边法向量方向上的投影区间不重叠，就说明两者不相交。若还要求交集多边形，可以用 Sutherland-Hodgman，把一个多边形依次按另一个凸多边形每条边对应的内部半平面裁剪。因为凸多边形本身就是这些半平面的交，所以裁完所有边后，剩下的就是交集多边形。

## 延伸阅读

- 任意多边形面积公式：[任意多边形面积公式](../polygon_area/README.md)
- 配套代码：[minimal.py](minimal.py)
