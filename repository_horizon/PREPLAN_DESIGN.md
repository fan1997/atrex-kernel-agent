# Repository Horizon Preplan 设计方案

## 1. 目标与定位

Preplan 解决的不是“怎样修改当前 kernel”，而是“在给定公开 contract 后，应选择什么端到端实现架构”。它位于正式 episode 之前，负责外层路线搜索；正式 episode 只负责选定路线内部的 kernel 实现和微观调优。

目标流程必须同时满足：

- 扩展搜索空间，不把 incumbent 的表示、算子边界或融合方式误认为硬约束；
- 允许静态源码研究、真实 GPU probing 和小规模原型；
- 用公开 contract 构造证据，不接触隐藏 shape、private evaluator 或 ABBA；
- 产出多路线 frontier，而不是过早收敛到单一路线；
- 用确定性 supervisor 校验边界、成本数据和产物结构；
- 不修改候选源码、不创建 candidate、不进入正式优化 episode。

当前 V1 Preplan 已证明这种流程能从 paged-KV 的结构性阻力出发，发现 paged-to-dense gather 再调用 dense FA 的 Mudi 大方向。但 V1 在初始 prompt 中直接暴露了四类路线名称，因此更准确的结论是“通用架构提示下的自主机制发现”，而不是完全无提示的独立重发现。下一版应采用“自由搜索后再分类”的流程。

## 2. Contract 的规范化表示

首先将公开 contract 规范化为：

\[
C=(S,X,I,L,H,P)
\]

- \(S\)：数学语义、精度与输出要求；
- \(X\)：公开输入域及其关系约束；
- \(I\)：输入输出接口与可观察行为；
- \(L\)：数据生命周期，是否允许预处理、缓存或跨调用复用；
- \(H\)：目标硬件及资源约束；
- \(P\)：production mode、框架、依赖和代码政策。

每项事实必须归入以下三类之一：

1. **硬约束**：由 contract、硬件或 policy 明确规定，不可改变；
2. **继承实现选择**：incumbent 当前采用，但 contract 没有强制；
3. **未验证假设**：模型暂时相信，但需要证据或实验确认。

搜索空间来自第二、第三类，而不是来自对硬约束的改写。

## 3. 将问题定义为双层实现图优化

Preplan 的搜索对象不是某个 kernel 文件，而是一张端到端实现图 \(G\)：

```text
input representation
  -> optional transforms
  -> one or more compute stages
  -> optional combine/postprocess
  -> output
```

一条路线的端到端成本定义为：

\[
T(G,\theta,x)=T_{transform}+\sum_iT_{stage_i}+T_{sync}+T_{dispatch}+T_{post}
\]

其中 \(G\) 是高层架构，\(\theta\) 是 tile、warp、pipeline stage 等实现参数。完整问题是：

\[
\min_{G\in\mathcal G(C)}\quad
\min_{\theta\in\Theta(G)}
\Phi_{x\in X}(T(G,\theta,x))
\]

并满足：

\[
Correct(G,\theta,x)=1,\qquad
Memory(G,\theta,x)\le M_{max},\qquad
Policy(G,\theta)=1
\]

- 外层 \(G\) 由 Preplan 搜索；
- 内层 \(\theta\) 由正式 episode 优化。

如果 contract 没有公开真实 shape 频率，不应猜测隐藏分布。路线比较应同时报告公开域上的聚合延迟、最坏退化、显存开销和适用区间，并保留 Pareto frontier，而不是伪造单一期望值。

## 4. 先寻找结构性阻力

在生成方案前，Preplan 必须先建立 workload/cost model：

- 计算量、访存量和算术强度；
- 输入中的间接寻址、稀疏性、mask 和 padding；
- 可利用的数据复用和跨 head/request 共享；
- 能暴露出的 CTA、warp 和 cluster 并行度；
- 哪些条件阻止 TMA、连续 tile 或已有 fast primitive；
- 当前接口是否允许预处理、缓存、额外 workspace 或多 launch；
- incumbent 失败是语义缺失、能力缺失还是单纯调度低效。

核心问题应抽象为：

> 哪项结构性阻力使 contract 无法映射到硬件或仓库中已有的快速路径？付出什么代价可以移除该阻力？

以当前 FA4 为例，结构性阻力不是泛泛的“attention kernel 不够快”，而是 HD256/page64/seqused-k 组合无法进入锁定源码现有的 paged fast path；paged-to-dense 是移除该阻力的一种图变换。

## 5. 自由路线生成：不提前暴露 taxonomy

第一阶段 prompt 不再出现以下四个名称：

- `direct_native`
- `representation_transform`
- `multi_stage`
- `hybrid_dispatch`

模型只获得 contract normal form、成本模型和可改变的决策轴，并被要求生成至少若干条“机制实质不同”的路线。推荐采用形态矩阵描述搜索空间：

- 数据表示：原生、连续、tile-packed、索引重映射、物化转换；
- 数据生命周期：调用内转换、流水重叠、跨调用缓存；
- 算子分解：全融合、分阶段、split/reduce、层次化流水；
- 算法：等价代数形式、online/reduction 结构、精确跳过无效工作；
- 并行映射：query、head、KV range、request、persistent worker、cluster；
- 数据移动：普通 load、TMA、shared staging、L2 reuse、multicast；
- 数值路径：scale 放置、累加精度、rescale 和重计算；
- 调度与选择：静态、动态队列、编译期变体、公开条件 dispatch；
- 资源权衡：临时显存、复制量、launch、寄存器、shared memory。

一条路线是这些轴上的一个一致决策向量，而不是一个标签。模型必须解释组合为何自洽，禁止为了数量机械拼接互不兼容的选择。

## 6. 路线的证明责任

每条自由生成路线都必须形成 route card：

- 稳定 route id；
- 核心 thesis；
- 实现图及数据表示；
- 必付转换、同步、launch 和 workspace 成本；
- 被解锁的 fast path；
- 必需硬件/软件机制；
- 理论 lower bound 与可能达到的性能区间；
- 预计胜出和失败的公开输入区域；
- 当前证据、反对证据和未验证假设；
- 最便宜的证伪实验；
- continue/kill 条件；
- 若失败，可转移到哪些 hedge route。

粗糙 prototype 的当前性能不能直接否定架构。Prototype 只用于证明语义、可实现性、固定开销或关键机制。只有满足以下条件之一才能淘汰架构：

- 即使按硬件 lower bound 估算，也不可能超过 incumbent；
- contract 或 policy 明确禁止必需机制；
- 关键语义无法满足；
- 多个机制等价但实现不同的 prototype 均证伪同一必要条件。

## 7. Probe 的选择与边界

Probe 的目标是减少路线排序的不确定性，而不是提前开始完整优化。下一项实验近似选择为：

\[
p^*=\arg\max_p
\frac{I(outcome_p;\ ranking(routes))}{Cost(p)}
\]

优先级依次是：

1. 能一次淘汰多条路线的语义/能力实验；
2. 决定路线是否可能胜出的强制成本；
3. 决定 crossover 的关键参数；
4. 仅改善局部实现的微观调参，留给正式 episode。

允许：

- 读取公开 contract、incumbent 和 bounded corpus；
- 通过 gateway 做公共合成输入的 GPU probe；
- 在 ignored `profiles/preplan/` 中创建有限小原型；
- 测量转换成本、launch floor、带宽、已有 primitive 能力和粗略区间。

禁止：

- 修改 manifest 声明的 editable source roots；
- 创建或提交候选源码；
- 调用 private evaluator、ABBA 或读取隐藏 shape；
- 使用外部 PR、未声明源码、Wiki 或网络搜索答案；
- 因 prototype 未优化而错误淘汰架构。

Probe 必须保存命令、公开输入描述、原始输出、单位、环境身份和解释。只在聊天中看到结果而未持久化，不算有效证据。

## 8. 后置 taxonomy 审计

自由 frontier 写入并封存后，supervisor 或独立 reviewer 才能看到四类 taxonomy，并对已有路线做后置映射：

- 原生直接处理；
- 表示转换；
- 多阶段分解；
- 混合 dispatch。

这些名称只用于发现覆盖缺口，不能回写或重解释第一阶段的自由发现记录。如果存在缺口，启动独立 gap-fill session；该 session 的新增路线必须单独标记为“taxonomy-guided”，不能与自由发现混为一谈。

这样既保留四类模式对搜索覆盖率的价值，也能区分真正自主发现和提示诱导。

## 9. 确定性比较与 portfolio

模型负责提出路线、机制和假设；supervisor 负责确定性计算和验证：

- 统一解析 probe 测量及单位；
- 计算端到端成本分解；
- 计算相对 incumbent 的收益 \(\Delta_r(x)\)；
- 标记路线的公开 win/loss region；
- 检查显存、正确性和 policy；
- 计算 Pareto dominance；
- 验证证据文件和 route card 的引用闭包。

最终不能只输出一个 winner，而应形成：

- primary route；
- 至少一条机制不同的 hedge route；
- 各路线的适用区间和不确定性；
- 下一批决定性实验；
- 明确的 continue/kill 条件；
- 可以并发拉起的隔离 implementation sessions。

`hybrid_dispatch` 只有在至少两条已实现路线分别在不同公开区间胜出后才成立；否则它只是额外开销，不能作为空泛路线占位。

## 10. 分阶段产物

建议将 Preplan 拆成不可回写的阶段产物：

1. `plans/contract_normal_form.json`
2. `plans/structural_cost_model.json`
3. `plans/free_architecture_frontier.json`
4. `profiles/preplan/` 与 `plans/probe_results.json`
5. `plans/taxonomy_audit.json`
6. `plans/route_portfolio.json`
7. `plans/preplan_run.json`

每个阶段由 supervisor 校验后封存，后续阶段只能引用，不能静默改写早期结论。最终 `preplan_run.json` 应记录使用的模型、session、源码 revision、全部产物 digest、是否接触 taxonomy、是否启动正式 episode以及机械验收结果。

## 11. 与正式优化的交接

Preplan 结束后不直接修改代码。每条入选路线应生成 implementation charter：

- 固定 thesis 和语义约束；
- 第一个必须实现的关键机制；
- 可以灵活改变的 loader、layout、schedule 和 pipeline；
- milestone、验证方法和 kill 条件；
- 与其他路线隔离的 workspace/session。

允许 primary 和 hedge route 并发探索，避免单个长 session 被首个方向锚定。正式 episode 的成功标准是实现路线所需机制并通过公开开发验证；最终晋升仍由 private ABBA 和完整 shape coverage 决定。

## 12. 自主发现能力的验证

若要宣称模型自主发现了某条已知路线，至少应进行以下消融：

- 初始 prompt 不出现该路线名称、关键 primitive 或 taxonomy；
- 使用全新 session，不继承旧计划和聊天记录；
- 不暴露目标 PR、实现源码和隐藏 benchmark 信息；
- 只允许 contract、锁定源码和正常 bounded corpus；
- 使用多个 seed 或模型重复；
- 统计自由 frontier 中是否出现相同的结构性图变换；
- 区分“发现架构方向”“发现具体机制”和“完成生产实现”三个层次。

当前 V1 结果可以表述为：模型在通用但包含 `representation_transform` 分类提示的 Preplan 中，自主定位了 page64/seqused-k 的结构性阻力，选择并验证了 paged-to-dense segmented gather，再复用 dense HD256 FA。它是有效的 guided discovery，但不能作为完全无提示重发现的最终证据。

## 13. 实施优先级

建议按以下顺序演进现有 V3 Preplan：

1. 修复 schema validator，使结构错误返回 violation 而不是抛出异常；
2. 将四类名称从初始 prompt 移至封存后的 taxonomy audit；
3. 增加 contract normal form 和 structural cost model；
4. 强制保存 probe 原始结果及 digest，禁止事后依赖聊天转述；
5. 将路线成本、win region 和 Pareto 比较移入确定性 supervisor；
6. 生成 primary/hedge implementation charter；
7. 支持多路线隔离 session 并发实现；
8. 用无 taxonomy prompt 的 Mudi 消融实验验证自主发现能力。

这一设计的核心不是让模型列出更多想法，而是把“从 contract 到实现”的过程拆成可审计的外层图搜索、最小证伪实验和内层实现优化，避免 incumbent 边界、首个实现和粗糙 prototype 过早收缩搜索空间。
