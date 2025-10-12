# 🤔 推理也要“激活值”吗？——显存账本（修正版 v1.1）

> **结论先行**  
> - 推理时也需要“激活/工作区（workspace）”内存（例如 Q/K/V、gate_up、临时输出、残差等）；  
> - 但**不会像训练那样跨层保留**用于反传，通常**只在当前层生命周期内驻留**，并被**复用**；  
> - **Prefill**（整段 prompt）阶段的激活峰值显著大于 **Decode**（步长=1）；  
> - 推理显存**真正的大头**通常是 **KV-Cache**，不是激活；  
> - 现代引擎会为 Prefill **预留一块工作区**，并用 **分块 Prefill**、**FlashAttention** 等手段把峰值压低。

---

## 目录
1. [训练 vs 推理：激活值差异](#1-训练-vs-推理激活值差异)  
2. [推理时到底在存什么](#2-推理时到底在存什么)  
3. [显存构成一览](#3-显存构成一览)  
4. [正确的内存估算方法](#4-正确的内存估算方法)  
5. [Prefill vs Decode：量级对比](#5-prefill-vs-decode量级对比)  
6. [SGLang/主流引擎的做法](#6-sglang主流引擎的做法)  
7. [常见误解与避坑](#7-常见误解与避坑)  
8. [一页纸速查表](#8-一页纸速查表)  

---

## 1) 训练 vs 推理：激活值差异

**训练**  
- 需要反向传播，框架会**跨层**保留大量中间张量（层输入、层输出、Norm 统计、激活前值等）。  
- 峰值 ≈ **所有层**中间状态的叠加，显存消耗巨大。  

**推理**  
- 无反传，**不跨层**保留激活。  
- 通常在**一层内**完成（Attn → 残差/Norm → MLP）后，即**复用同一块工作区**给下一层。  
- 峰值 ≈ **单层**在当前序列长度上的最大中间张量之和（而非乘以层数）。

> **要点**：推理的激活峰值≈“**单层峰值**”，而不是“按层数 × 叠加”。

---

## 2) 推理时到底在存什么

以 Decoder-only 层为例，在**当前层**计算期间会短暂驻留：
- **Q/K/V 投影输出**（含 RoPE 前/后）  
- **注意力内核工作区**（例如 FlashAttention 的块内缓冲；不 materialize `L×L`）  
- **MLP 的 `gate_up` 拼接张量**（`[*, 2I]`，往往是**最大**的那块）  
- **层内残差/Norm 融合缓冲**（可能与输出共享/就地写回）  
- **层输出**（下一层输入）

离开该层后，这些**被复用**给下一层；**不会跨层堆积**。

---

## 3) 显存构成一览

| 模块 | 规模关系 | 备注 |
|---|---|---|
| **权重** | 固定 | 可在 HBM；可配合权重量化/CPU offload（代价：带宽/延迟） |
| **KV-Cache** | **O(layers × tokens × (n_kv × d) × bytes)** | **大头**；与层数、总长度 T、批量 B 线性；可量化/分页 |
| **激活/工作区** | 取决于 **当前层**和 **当前 chunk 长度** | 峰值≈单层峰值；Prefill ≫ Decode |
| **图捕获/元数据** | 小~中等 | CUDA Graph capture、后端临时缓冲、调度元数据等 |

---

## 4) 正确的内存估算方法

> 以下估算假设：**使用 FlashAttention/块流式**，不 materialize `L×L` 注意力矩阵；数据类型以 **bf16/FP16（2 bytes）** 为例。  
> 记号：  
> - \(B\)：批大小（Prefill 中是并发序列数；Decode 中是活动请求数）  
> - \(L_{\text{chunk}}\)：Prefill 的**分块长度**（无分块时即为 L）  
> - \(H\)：hidden size  
> - \(I\)：intermediate size（MLP 上投后维度）  
> - \(n_{kv}\)：KV 头数（GQA 下 \(n_q > n_{kv}\)）  
> - \(d\)：每头维度（通常 \(H / n_q\)）  
> - \(\text{bytes}\)：元素字节数（bf16/FP16=2）  
> - \(c_H\)：覆盖 QKV/out-proj/残差等其他并存 H 级别张量的系数，经验 \(1\sim3\)

### 4.1 Prefill 激活峰值（单层主导项）
\[
M_{\text{act, prefill}}^{\text{peak}} \;\approx\; B \cdot L_{\text{chunk}} \cdot \big(2I \;+\; c_H\cdot H\big) \cdot \text{bytes}
\]
- **主导项是 `gate_up` 的 `2I` 拼接**；其余（QKV、Out、残差）以 \(c_H\cdot H\) 粗略覆盖。  
- **全模型峰值 ≈ 单层峰值**（层间复用工作区；除非 PP 并发/特殊融合增加并存）。

### 4.2 Decode 激活峰值（单步）
\[
M_{\text{act, decode}}^{\text{step}} \;\approx\; B \cdot \big(2I \;+\; c_H\cdot H\big) \cdot \text{bytes}
\]
- 与 \(L\) **无关**（步长=1），但与 **B 成正比**。  
- 通常≪ KV-Cache，但 B 很大时也不可忽略。

### 4.3 KV-Cache（真正的大头）
\[
M_{\text{KV}} \;\approx\; \text{layers} \cdot T \cdot B \cdot \big(2 \cdot n_{kv} \cdot d\big) \cdot \text{bytes}
\]
- \(T\)：总长度（prompt + 已生成）。  
- 若采用 **KV 量化**（FP8/INT8/INT4），\(\text{bytes}\) 按相应字节数缩放。  
- 采用 **分页 KV（paged KV）** 管理碎片与回收。

---

## 5) Prefill vs Decode：量级对比（举例）

> 以“7B 量级”模型近似：  
> \(H=4096\), \(I=11008\), \(n_q=32\), \(n_{kv}=8\)（GQA 比例 4），\(d = H/n_q = 128\)，bf16（2 bytes）

### 5.1 Prefill（B=1）
- **无分块**、处理 \(L=4096\)：  
  \[
  M_{\text{act, prefill}}^{\text{peak}} \approx 1 \cdot 4096 \cdot (2\cdot11008 + 2\cdot4096) \cdot 2
  \approx \mathbf{200\text{~MB}} \ \text{量级}
  \]
  （取 \(c_H=2\) 作为保守覆盖；与实现细节/融合程度有关，\~170–250 MB 合理）
- **分块 Prefill** \(L_{\text{chunk}}=2048\)：  
  峰值≈**减半**，\~100 MB 量级。  
  > 代价：需要多次前传，首 token 延迟小幅上升。

### 5.2 Decode（单步，B=64）
\[
M_{\text{act, decode}}^{\text{step}} \approx 64 \cdot (2\cdot11008 + 2\cdot4096)\cdot 2
\approx \mathbf{4\text{~MB}} \ \text{量级/层峰值}
\]
- **全模型峰值**≈单层峰值（层间复用）。  
- 相比 Prefill 激活小很多，但 B 大时并非“近零”。

### 5.3 KV-Cache（最主要）
- **每 token 每层**：
  \[
  2\cdot n_{kv}\cdot d \cdot \text{bytes} = 2\cdot 8 \cdot 128 \cdot 2 = \mathbf{4096\ \text{bytes}} \ (=4\text{~KB})
  \]
- **单请求，总长度 \(T=8000\)**、28 层：
  \[
  28 \cdot 8000 \cdot 4\text{~KB} \approx \mathbf{~0.86\ \text{GiB}}
  \]
- **并发 B** 线性放大；KV 量化按字节数线性缩放（FP8≈半、INT4≈四分之一）。

> **结论**：真实部署里，**KV-Cache 通常主导总显存**；Prefill 激活次之；Decode 激活更小。

---

## 6) SGLang/主流引擎的做法

- **预留激活工作区**：启动时根据上限（如 `max_prefill_tokens` 或 `chunked_prefill_size`、后端需求）**预留一块缓冲**，避免频繁 `cudaMalloc`。  
  > **注意**：这类预留是**经验性/后端相关**的上界，不是物理恒等式；不同模型（H/I/GQA）、不同注意力后端（FA2/FA3/自研 fused kernel）、是否图捕获都会影响峰值。
- **分块 Prefill**：把长 prompt 切成 2k/4k 等块，**线性**降低峰值激活与一次性 KV 写入压力。  
- **分页 KV（Paged KV）**：以固定大小页管理 KV，减少扩容拷贝与碎片；完成后可**回收页**。  
- **连续批处理（P+D 同步打包）**：Prefill 批与 Decode 批共同上卡，提高吞吐；Decode 批大小 \(B\) 会线性影响**当步激活**。  
- **KV 量化**：FP8/INT8/INT4 显著降低 KV 占用，通常是**最有效的显存优化**之一。  
- **FlashAttention / 块流式**：避免 `L×L` 显式矩阵，Prefill 激活从 \(O(L^2)\) 降到近似 \(O(L)\)。

---

## 7) 常见误解与避坑

- ❌ **“推理激活=按层数相乘”**  
  → ✅ 峰值≈**单层峰值**（层间复用），除非 PP 并发/特殊融合造成层间并存。  
- ❌ **“Decode 激活几乎为 0”**  
  → ✅ 单请求确小，但**随 B 线性增长**；B=64/128 时达到**数 MB 量级**并不罕见。  
- ❌ **“只缓存底层 KV，供全层复用”**  
  → ✅ KV 是**分层定义**的表征，各层 K/V 不可跨层混用。  
- ⚠️ **不使用 FlashAttention**  
  → Prefill 若 materialize `L×L`，激活暴涨为 \(O(L^2)\)，长序列会直接“炸显存”。  
- ⚠️ **忽视 GQA 比例**  
  → KV-Cache 规模与 \(n_{kv}\) 成正比；GQA 的 \(n_q:n_{kv}\) 比例显著影响 KV 占用与读带宽。

---

## 8) 一页纸速查表

**Prefill 激活（单层峰值）**  
\[
M \approx B \cdot L_{\text{chunk}} \cdot (2I + c_H H)\cdot \text{bytes}
\]
**Decode 激活（单步，单层峰值）**  
\[
M \approx B \cdot (2I + c_H H)\cdot \text{bytes}
\]
**KV-Cache**  
\[
M \approx \text{layers} \cdot T \cdot B \cdot (2 n_{kv} d)\cdot \text{bytes}
\]

**优化优先级**  
1) **KV 量化 / 分页 KV / 回收**  
2) **分块 Prefill（合适的 `L_chunk`）**  
3) **使用 FlashAttention/fused 内核**  
4) **控制 Decode 批大小 B**（在吞吐与峰值显存间权衡）  
5) **权重量化/Offload**（结合带宽预算）

---

**版本**：v1.1（2025-10-12）  
**说明**：上述估算为工程近似，实际峰值取决于具体实现（内核融合、图捕获、后端、dtype、对齐/填充等）。建议在目标引擎上配合 `nvidia-smi`、Nsight Systems/Compute、内存统计日志做**实测校准**。
