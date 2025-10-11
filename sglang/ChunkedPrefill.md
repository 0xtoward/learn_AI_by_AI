# SGLang Chunked Prefill 完全指南

> 基于 SGLang 源码的深度解析

## 📚 目录

1. [什么是 Chunked Prefill](#1-什么是-chunked-prefill)
2. [Chunked Prefill vs PD 分离](#2-chunked-prefill-vs-pd-分离)
3. [混合调度策略（P+D）](#3-混合调度策略pd)
4. [资源分配与比例](#4-资源分配与比例)
5. [触发条件与判断逻辑](#5-触发条件与判断逻辑)
6. [核心架构与文件](#6-核心架构与文件)
7. [完整工作流程](#7-完整工作流程)
8. [实战示例](#8-实战示例)

---

## 1. 什么是 Chunked Prefill

### 1.1 基本概念

**Chunked Prefill** 是一种将长输入序列的 Prefill 阶段分成多个小块（chunks）逐步处理的优化技术。

```
传统 Prefill:
输入 10,000 tokens → 一次性处理 → 生成 KV cache

Chunked Prefill:
输入 10,000 tokens → 分 5 个 chunk，每个 2,048 tokens
  ├─ Chunk 1: tokens[0:2048]
  ├─ Chunk 2: tokens[2048:4096]
  ├─ Chunk 3: tokens[4096:6144]
  ├─ Chunk 4: tokens[6144:8192]
  └─ Chunk 5: tokens[8192:10000]
```

### 1.2 为什么需要？

LLM 推理分为两个阶段：

1. **Prefill 阶段**：处理所有输入 tokens，计算 KV cache（计算密集）
2. **Decode 阶段**：逐个生成输出 tokens（内存密集）

**传统问题**：
- ❌ 长输入的 prefill 占用大量 GPU 内存（激活内存）
- ❌ Prefill 计算量大，其他请求等待时间长
- ❌ 无法与 decode 请求混合调度

**Chunked Prefill 优势**：
- ✅ 降低内存峰值：避免一次性分配大量激活内存
- ✅ 更好的批处理：可与 decode 请求混合调度
- ✅ 降低延迟：其他请求不需要等待长 prefill 完成
- ✅ 利用 Radix Cache：已处理的 chunk 可被其他请求复用

### 1.3 核心代码位置

**判断是否分块** (`schedule_policy.py:593-638`):

```python
if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
    # Non-chunked prefill - 直接完整处理
    self.can_run_list.append(req)
    # ...
else:
    # Chunked prefill - 分块处理
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
    
    # 设置本次处理的长度
    req.extend_input_len = trunc_len
    req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
    
    self.can_run_list.append(req)
    self.new_chunked_req = req  # 保存为 chunked_req，下次继续
```

**缓存未完成的请求** (`radix_cache_cpp.py:185-230`):

```python
def cache_unfinished_req(self, req: Req, chunked=False):
    """Cache request when it is unfinished."""
    token_ids = req.fill_ids
    prefill_len = len(token_ids)  # prefill only (maybe chunked)
    kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :prefill_len]
    
    # 插入到 Radix Tree
    old_prefix_len = len(req.prefix_indices) // self.page_size * self.page_size
    new_prefix_len = self._insert(RadixKey(token_ids, req.extra_key), kv_indices)
    
    # 重新匹配前缀，复用已有的 KV cache
    new_indices_vec, _, new_last_node, _ = self.tree_cache.match_prefix(
        RadixKey(token_ids, req.extra_key).token_ids
    )
    new_indices = self._merge_tensor(new_indices_vec)
    
    # 更新请求的前缀索引
    if old_prefix_len < new_prefix_len:
        self.token_to_kv_pool.free(kv_indices[old_prefix_len:new_prefix_len])
        reused_indices = new_indices[old_prefix_len:new_prefix_len]
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, old_prefix_len:new_prefix_len
        ] = reused_indices
    
    req.prefix_indices = new_indices
    req.last_node = new_last_node
```

---

## 2. Chunked Prefill vs PD 分离

这是两种**完全不同**的优化技术，解决不同的问题。

### 2.1 对比表

| 维度 | **PD 分离** | **Chunked Prefill + Mixed** |
|------|------------|----------------------------|
| **设计理念** | 物理分离 | 逻辑统合 |
| **部署架构** | 两个独立集群 | 单一集群 |
| **GPU 使用** | 分开的 GPU | 相同 GPU |
| **数据传输** | ✅ 需要（网络/RDMA） | ❌ 不需要 |
| **调度器** | 两个独立调度器 | 一个统一调度器 |
| **Prefill 处理** | 完整 Prefill 一次处理 | 分块处理 |
| **混合执行** | ❌ 无法混合 | ✅ 同一 batch 混合 |
| **适用场景** | 大规模部署，资源充足 | 单机/资源受限 |
| **延迟** | 有网络传输开销 | 更低（本地） |
| **吞吐** | 更高（专门优化） | 中等 |

### 2.2 PD 分离：物理分离

```
┌─────────────────┐      KV Cache      ┌─────────────────┐
│  Prefill 集群    │ ─────传输────────> │  Decode 集群     │
│  GPU 0, 1, 2    │   (Mooncake/NIXL)  │  GPU 3, 4, 5    │
│  专门做 Prefill  │                     │  专门做 Decode   │
└─────────────────┘                     └─────────────────┘

特点：
- 两个完全独立的 event loop
- 需要通过网络传输 KV cache
- Prefill 和 Decode 互不干扰
```

**代码位置** (`disaggregation/prefill.py`, `disaggregation/decode.py`)

**启动示例**:
```bash
# Prefill 服务器
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0

# Decode 服务器  
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B \
  --disaggregation-mode decode \
  --port 30001 \
  --disaggregation-ib-device mlx5_0
```

### 2.3 Chunked Prefill + Mixed：逻辑统合

```
┌────────────────────────────────────┐
│        单一 Scheduler (同一 GPU)    │
├────────────────────────────────────┤
│  Batch 1 = [Prefill Chunk 2048t,   │
│             Decode Req A (1t),     │
│             Decode Req B (1t)]     │
│                                    │
│  同一个 forward 调用处理！          │
└────────────────────────────────────┘

特点：
- 单一 event loop
- 所有数据在本地 GPU
- Prefill chunks 和 Decode 在同一 batch
```

**代码位置** (`scheduler.py:1996-2014`)

**启动示例**:
```bash
# 单一服务器，启用混合调度
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B \
  --chunked-prefill-size 4096 \
  --enable-mixed-chunk  # 🔥 关键参数
```

---

## 3. 混合调度策略（P+D）

Chunked Prefill 通过**分块**使得 Prefill 变得"轻量"，从而可以和 Decode 在同一个 batch 中执行。

### 3.1 核心思想

```python
# 问题：如何在一个 batch 中同时处理 Prefill 和 Decode？
# 答案：通过 extend_lens 告诉 attention 层每个请求的处理长度

Batch = {
    reqs: [Prefill_Req, Decode_A, Decode_B],
    input_ids: [p0, p1, ..., p2047, dA, dB],  # 2050 个 tokens
    extend_lens: [2048, 1, 1],  # 🔑 关键！
    #            ^^^^  ^  ^
    #         Prefill  D  D
    #                  A  B
}
```

### 3.2 三步混合策略

#### Step 1: 资源预留（在调度时）

**代码位置** (`scheduler.py:1871-1881`):

```python
# 创建 PrefillAdder 时预留 decode 空间
adder = PrefillAdder(
    self.page_size,
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    self.running_batch,
    self.new_token_ratio,
    self.max_prefill_tokens,
    self.chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0,  # 🔑 预留空间
    #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #            如果启用混合，减去 running batch 的大小
    self.priority_scheduling_preemption_threshold,
)
```

**内部逻辑** (`schedule_policy.py:333-336`):

```python
self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
self.rem_chunk_tokens = rem_chunk_tokens
if self.rem_chunk_tokens is not None:
    self.rem_chunk_tokens -= mixed_with_decode_tokens
    
# 例如：
# chunked_prefill_size = 2048
# mixed_with_decode_tokens = 10 (有 10 个 decode 请求)
# 实际可用于 prefill: 2048 - 10 = 2038 tokens
```

#### Step 2: 判断是否分块（添加 Prefill 请求时）

**代码位置** (`schedule_policy.py:593-638`):

```python
if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
    # 场景 1: 输入很短，不需要分块
    # 例如：input_tokens=100, rem_chunk_tokens=2038
    # 100 < 2038，直接完整处理
    self.can_run_list.append(req)
    self._update_prefill_budget(prefix_len, input_tokens, max_new_tokens)
else:
    # 场景 2: 输入很长，需要分块
    # 例如：input_tokens=10000, rem_chunk_tokens=2038
    # 10000 > 2038，分块处理
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
    req.extend_input_len = trunc_len  # 本次只处理 2038 tokens
    req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
    
    self.can_run_list.append(req)
    self.new_chunked_req = req  # 保存，下次继续处理剩余部分
```

#### Step 3: 物理混合（创建 Batch 时）

**代码位置** (`scheduler.py:1996-2014`):

```python
# 创建新的 prefill batch
new_batch = ScheduleBatch.init_new(can_run_list, ...)
new_batch.prepare_for_extend()

# Mixed-style chunked prefill
if (
    self.is_mixed_chunk
    and not self.running_batch.is_empty()
    and not (new_batch.return_logprob or self.running_batch.return_logprob)
):
    # 过滤掉已完成的 decode 请求
    self.running_batch.filter_batch()
    if not self.running_batch.is_empty():
        # 准备 decode batch
        self.running_batch.prepare_for_decode()
        
        # 🔥 关键：混合！
        new_batch.mix_with_running(self.running_batch)
        new_batch.decoding_reqs = self.running_batch.reqs
    
    # 清空 running_batch
    self.running_batch = ScheduleBatch(
        reqs=[], batch_is_full=self.running_batch.batch_is_full
    )
else:
    new_batch.decoding_reqs = None

return new_batch
```

**混合函数** (`schedule_batch.py:1458-1486`):

```python
def mix_with_running(self, running_batch: "ScheduleBatch"):
    # 1. 设置混合模式
    self.forward_mode = ForwardMode.MIXED
    running_bs = running_batch.batch_size()
    
    # 2. 设置 decode 请求的 extend_len = 1
    for req in running_batch.reqs:
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.extend_input_len = 1  # 🔑 Decode 只处理 1 个 token
    
    # 3. 拼接张量
    input_ids = torch.cat([self.input_ids, running_batch.input_ids])
    #                      ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
    #                      Prefill tokens   Decode tokens
    out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])
    
    # 4. 合并 batch
    self.merge_batch(running_batch)
    self.input_ids = input_ids
    self.out_cache_loc = out_cache_loc
    
    # 5. 更新 extend_lens
    delta = 0 if self.enable_overlap else -1
    self.prefix_lens.extend([
        len(r.origin_input_ids) + len(r.output_ids) + delta
        for r in running_batch.reqs
    ])
    self.extend_lens.extend([1] * running_bs)  # 每个 decode 都是 1
    self.extend_num_tokens += running_bs
```

### 3.3 ForwardMode 的作用

**定义** (`forward_batch_info.py:62-70`):

```python
class ForwardMode(IntEnum):
    EXTEND = auto()   # 纯 Prefill
    DECODE = auto()   # 纯 Decode
    MIXED = auto()    # 🔥 Prefill + Decode 混合！
    IDLE = auto()
    TARGET_VERIFY = auto()
    DRAFT_EXTEND = auto()
```

Attention 层会根据 `forward_mode` 和 `extend_lens` 判断如何处理每个请求：

```python
# 伪代码：Attention 层的逻辑
if forward_mode == ForwardMode.MIXED:
    for i, extend_len in enumerate(extend_lens):
        if extend_len > 1:
            # Prefill attention: all-to-all
            attention_prefill(tokens[offset:offset+extend_len])
        else:
            # Decode attention: attend to history
            attention_decode(tokens[offset], history_kv_cache)
        offset += extend_len
```

---

## 4. 资源分配与比例

### 4.1 D 和 P 的资源占用

**答案：在调度预算时，确实都按 1 token 计算！**

```python
# schedule_policy.py:333-336
self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
if self.rem_chunk_tokens is not None:
    self.rem_chunk_tokens -= mixed_with_decode_tokens
    
# 直接从可用 token 预算中减去 decode 的数量，没有任何权重系数！
```

### 4.2 真实的 P:D Token 比例

从 `server_args.py` 的自动配置数据：

| GPU 类型 | Prefill Chunk Size | Max Decode Batch | 理论最大比例 | 
|---------|-------------------|------------------|-------------|
| **T4 / RTX 4080** | 2,048 | 8 | **256:1** |
| **RTX 4090 (TP1)** | 2,048 | 16 | **128:1** |
| **RTX 4090 (TP4+)** | 2,048 | 80 | **25:1** |
| **A100 40GB (TP1)** | 4,096 | 32 | **128:1** |
| **A100 40GB (TP4+)** | 4,096 | 160 | **25:1** |
| **H100 (TP1)** | 8,192 | 256 | **32:1** |
| **H100 (TP4+)** | 8,192 | 512 | **16:1** |
| **B200 / MI300** | 16,384 | 512 | **32:1** |

**代码位置** (`server_args.py:575-631`)

### 4.3 实际运行示例

```python
═══════════════════════════════════════════════════════
场景：RTX 4090 (TP1)
═══════════════════════════════════════════════════════
chunked_prefill_size = 2048
cuda_graph_max_bs = 16
running_batch 有 8 个 decode 请求

资源分配：
- Prefill tokens: 2048 - 8 = 2040
- Decode tokens: 8
- 实际比例: 2040:8 = 255:1

═══════════════════════════════════════════════════════
混合 Batch 组成：
═══════════════════════════════════════════════════════
new_batch = {
    reqs: [prefill_req, d1, d2, d3, d4, d5, d6, d7, d8],
    input_ids: [p0, p1, ..., p2039, d1, d2, d3, d4, d5, d6, d7, d8],
    extend_lens: [2040, 1, 1, 1, 1, 1, 1, 1, 1],
    forward_mode: ForwardMode.MIXED
}

总 token 数 = 2040 + 8 = 2048 ✓
```

### 4.4 对应关系

```
N 个 Decode 任务 → N × 1 token/任务 = N 个 tokens

例如：
- 2 个 decode 任务 → 2 个 tokens（每个任务 1 token）
- 10 个 decode 任务 → 10 个 tokens（每个任务 1 token）
- 100 个 decode 任务 → 100 个 tokens（每个任务 1 token）
```

每个 decode 请求在一次 forward 中只生成 1 个 token，这是 auto-regressive 生成的本质！

---

## 5. 触发条件与判断逻辑

### 5.1 核心判断逻辑

**只看输入 tokens 数量，不看输出！**

```python
# schedule_policy.py:593
if input_tokens <= self.rem_chunk_tokens:
    # 不分块 - 直接完整 prefill
else:
    # 分块 - Chunked prefill
```

### 5.2 场景分析

#### 场景 A: 长文档分析（会触发）

```python
prompt = "请分析这篇10万字的论文：[10万字内容]"
input_tokens ≈ 150,000 tokens
max_new_tokens = 2000
chunked_prefill_size = 2048

判断：
input_tokens (150,000) > chunked_prefill_size (2048) ✓

结果：✅ 触发 Chunked Prefill！
分成约 73 个 chunks (150,000 / 2048)
```

#### 场景 B: 多图片分析（会触发）

```python
prompt = "分析这50张图片" + [50张图片]
input_tokens ≈ 50 * 1000 = 50,000 tokens (每张图 ~1k tokens)
max_new_tokens = 500
chunked_prefill_size = 4096

判断：
input_tokens (50,000) > chunked_prefill_size (4096) ✓

结果：✅ 触发 Chunked Prefill！
分成约 12 个 chunks
```

#### 场景 C: 短 prompt + 长输出（**不**触发）

```python
prompt = "给我写个5000字的散文"
input_tokens ≈ 10 tokens
max_new_tokens = 7500
chunked_prefill_size = 2048

判断：
input_tokens (10) < chunked_prefill_size (2048) ✓

结果：❌ 不触发 Chunked Prefill
Prefill 一次完成，然后进入长时间的 Decode 阶段
```

### 5.3 设计哲学

Chunked Prefill 是为**长输入**场景设计的：
- 📄 长文档理解
- 🖼️ 多图片视觉任务
- 💬 超长对话历史

对于短输入长输出的生成任务，它不介入，让传统 continuous batching 自然工作。

### 5.4 完整流程示例

```python
═══════════════════════════════════════════════════════
场景：长文档分析
═══════════════════════════════════════════════════════
输入：150,000 tokens 的文档
chunked_prefill_size = 2048
running_batch = 10 个 decode 请求

Step 1: 第一轮调度
────────────────────────────────────────────────────────
rem_chunk_tokens = 2048 - 10 = 2038
input_tokens = 150,000
input_tokens > rem_chunk_tokens ✓ → 触发分块

处理：tokens[0:2038]
Batch = {
    reqs: [doc_req, d1, d2, ..., d10],
    extend_lens: [2038, 1, 1, ..., 1],
    forward_mode: MIXED
}
剩余：150,000 - 2038 = 147,962 tokens

Step 2: 第二轮调度
────────────────────────────────────────────────────────
rem_chunk_tokens = 2048 - 10 = 2038
剩余 input_tokens = 147,962
input_tokens > rem_chunk_tokens ✓ → 继续分块

处理：tokens[2038:4076]
剩余：147,962 - 2038 = 145,924 tokens

...重复约 73 轮...

Step 73: 最后一轮
────────────────────────────────────────────────────────
剩余 input_tokens = 1500
input_tokens < rem_chunk_tokens ✓ → 完成分块

处理：tokens[148,500:150,000]
chunked_req = None  # 完成

Step 74: 开始 Decode
────────────────────────────────────────────────────────
Prefill 完成，开始生成 2000 tokens 的输出
```

---

## 6. 核心架构与文件

### 6.1 整体架构

```
用户请求
   ↓
server_args.py (配置层)
   ↓
scheduler.py (调度核心)
   ├── schedule_policy.py (策略层)
   ├── schedule_batch.py (数据层)
   └── 循环调度
```

### 6.2 文件职责

#### 1. `server_args.py` - 配置中心

**作用**：定义 SGLang 启动的所有参数

**关键配置**：
```python
@dataclasses.dataclass
class ServerArgs:
    # Chunked Prefill 相关
    chunked_prefill_size: Optional[int] = None  # chunk 大小
    enable_mixed_chunk: bool = False            # 是否混合调度
    max_prefill_tokens: int = 16384            # 最大 prefill tokens
    
    # 自动配置
    def _handle_gpu_memory_settings(self, gpu_mem):
        if gpu_mem < 20 * 1024:
            self.chunked_prefill_size = 2048
        elif gpu_mem < 35 * 1024:
            self.chunked_prefill_size = 2048
        elif gpu_mem < 60 * 1024:
            self.chunked_prefill_size = 4096
        # ...
```

#### 2. `scheduler.py` - 调度核心

**作用**：SGLang 的"大脑"，负责整个推理流程

**核心方法**：
```python
class Scheduler:
    def event_loop_normal(self):
        """主事件循环"""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            batch = self.get_next_batch_to_run()
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
    
    def get_new_batch_prefill(self) -> ScheduleBatch:
        """创建 prefill batch（Chunked Prefill 入口）"""
        adder = PrefillAdder(...)
        
        # 处理上一轮的 chunked_req
        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)
        
        # 添加新请求
        for req in self.waiting_queue:
            adder.add_one_req(req, ...)
        
        # 创建 batch
        new_batch = ScheduleBatch.init_new(adder.can_run_list, ...)
        
        # 混合调度
        if self.is_mixed_chunk and not self.running_batch.is_empty():
            new_batch.mix_with_running(self.running_batch)
        
        return new_batch
```

#### 3. `schedule_batch.py` - 数据结构层

**作用**：定义请求和批次的数据结构

**核心类**：
```python
class Req:
    """单个请求"""
    origin_input_ids: List[int]  # 原始输入
    output_ids: List[int]        # 已生成输出
    prefix_indices: torch.Tensor # 缓存的 KV 索引
    extend_input_len: int        # 本次处理长度（Chunked Prefill）
    fill_ids: List[int]          # 当前需要填充的 IDs

class ScheduleBatch:
    """批次"""
    reqs: List[Req]
    forward_mode: ForwardMode    # EXTEND / DECODE / MIXED
    prefix_lens: List[int]       # 每个请求的前缀长度
    extend_lens: List[int]       # 每个请求的扩展长度
    decoding_reqs: List[Req]     # 混合模式下的 decode 请求
    
    def mix_with_running(self, running_batch):
        """混合 Prefill 和 Decode"""
        self.forward_mode = ForwardMode.MIXED
        for req in running_batch.reqs:
            req.extend_input_len = 1  # Decode 只处理 1 token
        self.input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        self.extend_lens.extend([1] * running_batch.batch_size())
```

#### 4. `schedule_policy.py` - 策略决策层

**作用**：决定请求的调度顺序和资源分配

**核心类**：
```python
class SchedulePolicy:
    """排序策略"""
    def calc_priority(self, waiting_queue: List[Req]):
        # FCFS, LPM, LOF, DFS_WEIGHT, RANDOM
        pass

class PrefillAdder:
    """资源预算管理（最重要！）"""
    def __init__(self, ..., mixed_with_decode_tokens: int = 0):
        # 预留 decode 空间
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens - mixed_with_decode_tokens
    
    def add_one_req(self, req: Req):
        """判断是否需要分块"""
        if input_tokens <= self.rem_chunk_tokens:
            # 不分块
            self.can_run_list.append(req)
        else:
            # 分块
            trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            req.extend_input_len = trunc_len
            self.new_chunked_req = req
    
    def add_chunked_req(self, req: Req):
        """继续处理分块请求"""
        req.extend_input_len = min(req.extend_input_len, self.rem_chunk_tokens)
        return req if truncated else None
```

### 6.3 关系图

```
┌─────────────────────────────────────────────────────┐
│                   server_args.py                     │
│  ┌────────────────────────────────────────────┐    │
│  │ ServerArgs                                  │    │
│  │ - chunked_prefill_size: 2048-16384         │    │
│  │ - enable_mixed_chunk: True/False           │    │
│  │ - cuda_graph_max_bs: 8-512                 │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│                   scheduler.py                       │
│  ┌────────────────────────────────────────────┐    │
│  │ Scheduler                                   │    │
│  │                                             │    │
│  │  event_loop_normal()                        │    │
│  │    ├─ recv_requests()                       │    │
│  │    ├─ get_next_batch_to_run()              │    │
│  │    │    └─ get_new_batch_prefill() ◄────┐  │    │
│  │    ├─ run_batch()                        │  │    │
│  │    └─ process_batch_result()             │  │    │
│  │                                           │  │    │
│  │  chunked_req: Req                        │  │    │
│  │  is_mixed_chunk: bool                    │  │    │
│  └───────────────────────────────────────┬──┘  │    │
└────────────────────────────────────────┬─┼─────┘    │
                                         │ │          │
                  ┌──────────────────────┘ │          │
                  ↓                        ↓          │
┌─────────────────────────────┐  ┌───────────────────┴──┐
│    schedule_policy.py       │  │  schedule_batch.py    │
│  ┌────────────────────┐     │  │  ┌──────────────────┐ │
│  │ SchedulePolicy     │     │  │  │ Req              │ │
│  │ - calc_priority()  │     │  │  │ - fill_ids       │ │
│  └────────────────────┘     │  │  │ - prefix_indices │ │
│                             │  │  │ - extend_input_len│ │
│  ┌────────────────────┐     │  │  └──────────────────┘ │
│  │ PrefillAdder       │     │  │                       │
│  │ - add_one_req()    │◄────┼──┤  ┌──────────────────┐ │
│  │ - add_chunked_req()│     │  │  │ ScheduleBatch    │ │
│  │ - rem_chunk_tokens │     │  │  │ - forward_mode   │ │
│  └────────────────────┘     │  │  │ - extend_lens    │ │
│                             │  │  │ - mix_with_running()│
└─────────────────────────────┘  │  └──────────────────┘ │
                                 └───────────────────────┘
```

---

## 7. 完整工作流程

### 7.1 单次 Chunked Prefill 流程

```python
# ========================================
# 1. 用户请求到达
# ========================================
用户发送: "分析这篇 10 万字的论文：[文本]"
→ Tokenizer: 生成 150,000 tokens

# ========================================
# 2. 创建 Req 对象
# ========================================
req = Req(
    rid="req_123",
    origin_input_ids=[token_0, token_1, ..., token_149999],  # 150k tokens
    sampling_params=SamplingParams(max_new_tokens=2000),
    ...
)
waiting_queue.append(req)

# ========================================
# 3. Scheduler 事件循环
# ========================================
def event_loop_normal():
    while True:
        # 3.1 获取下一个 batch
        batch = get_next_batch_to_run()
        
        # 3.2 如果需要 prefill
        if need_prefill:
            batch = get_new_batch_prefill()

# ========================================
# 4. 创建 Prefill Batch
# ========================================
def get_new_batch_prefill():
    # 4.1 排序策略
    policy.calc_priority(waiting_queue)
    
    # 4.2 创建 PrefillAdder
    running_bs = len(running_batch.reqs)  # 假设 10 个 decode
    adder = PrefillAdder(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=token_to_kv_pool,
        running_batch=running_batch,
        new_token_ratio=0.4,
        rem_input_tokens=16384,
        rem_chunk_tokens=2048,
        mixed_with_decode_tokens=10 if is_mixed_chunk else 0,  # 🔑
    )
    # adder.rem_chunk_tokens = 2048 - 10 = 2038
    
    # 4.3 处理之前的 chunked_req
    if chunked_req is not None:
        chunked_req.init_next_round_input()
        chunked_req = adder.add_chunked_req(chunked_req)
    
    # 4.4 添加新请求
    for req in waiting_queue:
        req.init_next_round_input(tree_cache)
        # 匹配前缀
        match_result = tree_cache.match_prefix(req.origin_input_ids)
        req.prefix_indices = match_result.device_indices  # 假设 0（无缓存）
        req.extend_input_len = len(req.origin_input_ids)  # 150,000
        
        # 判断是否分块
        result = adder.add_one_req(req, has_chunked_req=False, ...)
        # input_tokens=150,000 > rem_chunk_tokens=2038 → 分块！
        # req.extend_input_len = 2038
        # req.fill_ids = origin_input_ids[0:2038]
        # adder.new_chunked_req = req
        
        break  # 只添加一个请求
    
    # 4.5 创建 batch
    new_batch = ScheduleBatch.init_new(
        reqs=adder.can_run_list,  # [req]
        ...
    )
    new_batch.prepare_for_extend()
    # new_batch.input_ids = [token_0, ..., token_2037]
    # new_batch.extend_lens = [2038]
    
    # 4.6 混合调度
    if is_mixed_chunk and not running_batch.is_empty():
        running_batch.filter_batch()  # 过滤完成的请求
        running_batch.prepare_for_decode()
        
        new_batch.mix_with_running(running_batch)
        # new_batch.input_ids = [token_0, ..., token_2037, d1, d2, ..., d10]
        # new_batch.extend_lens = [2038, 1, 1, ..., 1]
        # new_batch.forward_mode = ForwardMode.MIXED
    
    return new_batch

# ========================================
# 5. 运行 Batch
# ========================================
result = run_batch(new_batch)
# GPU forward:
#   - Prefill: tokens[0:2038]
#   - Decode: 10 个请求各生成 1 token

# ========================================
# 6. 处理结果
# ========================================
process_batch_result(new_batch, result)
# 6.1 缓存 chunked_req
tree_cache.cache_unfinished_req(chunked_req, chunked=True)
# 6.2 更新 chunked_req
chunked_req.fill_ids = origin_input_ids  # 恢复完整输入
chunked_req.extend_input_len = 150,000 - 2038  # 剩余部分

# ========================================
# 7. 下一轮调度
# ========================================
# chunked_req 会在下一轮继续处理
# 重复步骤 4-6，直到所有 150k tokens 处理完毕
```

### 7.2 多轮 Chunked Prefill 时间线

```
时间线：
────────────────────────────────────────────────────────
Round 1:  [Prefill 2038t] [D1 D2 ... D10]  → 处理 tokens[0:2038]
            剩余：147,962 tokens
            
Round 2:  [Prefill 2038t] [D1 D2 ... D10]  → 处理 tokens[2038:4076]
            剩余：145,924 tokens
            
Round 3:  [Prefill 2038t] [D1 D2 ... D10]  → 处理 tokens[4076:6114]
            剩余：143,886 tokens
            
...重复约 73 轮...

Round 73: [Prefill 1500t] [D1 D2 ... D10]  → 处理 tokens[148,500:150,000]
            Prefill 完成！
            
Round 74: [D1 D2 ... D10]                  → 开始生成输出
Round 75: [D1 D2 ... D10]
...
Round 2073: [D1]                           → 最后一个请求完成
────────────────────────────────────────────────────────
```

### 7.3 关键代码调用链

```
1. scheduler.py:event_loop_normal()
   └─> 2. scheduler.py:get_next_batch_to_run()
       └─> 3. scheduler.py:get_new_batch_prefill()
           ├─> 4. schedule_policy.py:PrefillAdder.__init__()
           │   └─> rem_chunk_tokens -= mixed_with_decode_tokens
           │
           ├─> 5. schedule_policy.py:add_one_req()
           │   └─> if input_tokens > rem_chunk_tokens:
           │       ├─> req.extend_input_len = trunc_len
           │       └─> self.new_chunked_req = req
           │
           ├─> 6. schedule_batch.py:ScheduleBatch.init_new()
           │   └─> 创建 batch 对象
           │
           └─> 7. schedule_batch.py:mix_with_running()
               ├─> self.forward_mode = ForwardMode.MIXED
               ├─> req.extend_input_len = 1 (for decode)
               ├─> input_ids = torch.cat([prefill, decode])
               └─> extend_lens.extend([1] * running_bs)
```

---

## 8. 实战示例

### 8.1 启动配置

```bash
# 示例 1: RTX 4090 单卡，启用混合调度
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B \
  --chunked-prefill-size 2048 \
  --enable-mixed-chunk \
  --cuda-graph-max-bs 16 \
  --port 30000

# 示例 2: H100 单卡，更大的 chunk
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-70B \
  --chunked-prefill-size 8192 \
  --enable-mixed-chunk \
  --cuda-graph-max-bs 256 \
  --tp-size 4

# 示例 3: 禁用 chunked prefill
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B \
  --chunked-prefill-size -1  # -1 表示禁用
```

### 8.2 日志解读

```
# 启动日志
[2025-01-10 10:00:00] max_total_num_tokens=665690, chunked_prefill_size=2048, 
                       max_prefill_tokens=16384, max_running_requests=4096, 
                       context_len=65536, available_gpu_mem=13.50 GB

# Prefill batch 日志
[2025-01-10 10:01:00] Prefill batch. #new-seq: 1, #new-token: 2000, 
                       #cached-token: 0, token usage: 0.15, 
                       #running-req: 20, #queue-req: 5

解读：
- #new-seq: 1              → 1 个新请求
- #new-token: 2000         → 本次 prefill 处理 2000 tokens (chunked)
- #cached-token: 0         → 无缓存命中
- token usage: 0.15        → KV cache 使用率 15%
- #running-req: 20         → 有 20 个 decode 请求正在运行
- #queue-req: 5            → 等待队列中有 5 个请求

实际比例: 2000 prefill : 20 decode = 100:1

# Decode batch 日志
[2025-01-10 10:01:01] Decode batch. #running-req: 21, #token: 12500, 
                       token usage: 0.18
```

### 8.3 性能对比

```python
# 场景：处理 100k token 的长文档

# 不使用 Chunked Prefill
────────────────────────────────────────────────────────
时间    操作
0s      Prefill 100k tokens (阻塞所有其他请求)
15s     Prefill 完成，开始 Decode
        其他 20 个请求在等待队列中饿死
        TTFT (Time To First Token): 15s
────────────────────────────────────────────────────────

# 使用 Chunked Prefill (2048 chunk)
────────────────────────────────────────────────────────
时间    操作
0s      Round 1: Prefill 2048t + Decode 20 reqs
0.3s    Round 2: Prefill 2048t + Decode 20 reqs
0.6s    Round 3: Prefill 2048t + Decode 20 reqs
...
15s     Round 49: Prefill 完成，继续 Decode
        其他 20 个请求持续生成，没有饿死
        TTFT (平均): 0.5s (其他请求)
────────────────────────────────────────────────────────

性能提升：
- 其他请求的 TTFT: 15s → 0.5s (30x 改善)
- 吞吐量提升: 约 2-3x
- 内存峰值: 降低约 60%
```

### 8.4 最佳实践

```python
# 1. 根据 GPU 内存选择合适的 chunk size
GPU Memory < 24GB  → chunked_prefill_size=2048
GPU Memory 40GB    → chunked_prefill_size=4096
GPU Memory 80GB+   → chunked_prefill_size=8192

# 2. 根据工作负载选择是否启用混合调度
长输入 + 高并发    → --enable-mixed-chunk
短输入为主         → 不启用混合（默认）

# 3. 配合 Radix Cache 使用
--disable-radix-cache=False  # 启用 Radix Cache
→ 已处理的 chunks 可以被其他请求复用

# 4. 监控关键指标
- token usage: 应保持在 60-80%
- #running-req: 应接近 max_running_requests 的 70-80%
- TTFT (Time To First Token): 应 < 1s
- ITL (Inter Token Latency): 应 < 50ms
```

### 8.5 调试技巧

```python
# 1. 查看是否触发 Chunked Prefill
grep "Chunked prefill" /path/to/sglang.log

# 2. 查看 chunk 大小
grep "chunked_prefill_size" /path/to/sglang.log

# 3. 查看混合调度统计
grep "Prefill batch" /path/to/sglang.log | grep "#running-req"
# 如果 #running-req > 0，说明混合调度生效

# 4. 查看内存使用
grep "available_gpu_mem" /path/to/sglang.log

# 5. 使用 Python 客户端测试
import openai
client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.completions.create(
    model="meta-llama/Llama-3-8B",
    prompt="[很长的输入文本，150k tokens]",
    max_tokens=100,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].text, end="", flush=True)
```

---

## 总结

### 核心要点

1. **Chunked Prefill** 是将长输入分块处理的技术，降低内存峰值
2. **与 PD 分离不同**：PD 分离是物理分离，Chunked Prefill 是逻辑统合
3. **混合调度**：通过 `ForwardMode.MIXED` 和 `extend_lens` 实现 P+D 同 batch
4. **资源分配**：P 和 D 按 1 token 计算，实际比例约 16:1 到 256:1
5. **触发条件**：只看输入 tokens，不看输出（`input_tokens > chunked_prefill_size`）
6. **核心文件**：
   - `server_args.py`: 配置
   - `scheduler.py`: 调度核心
   - `schedule_policy.py`: 策略和预算
   - `schedule_batch.py`: 数据结构

### 适用场景

✅ **适合 Chunked Prefill**:
- 长文档理解
- 多图片/视频分析
- 超长对话历史
- 高并发服务

❌ **不适合 Chunked Prefill**:
- 短输入长输出（如代码生成）
- 低并发场景
- 追求极致吞吐量（用 PD 分离）

### 配置建议

```bash
# 推荐配置（平衡性能和延迟）
python -m sglang.launch_server \
  --model-path YOUR_MODEL \
  --chunked-prefill-size 4096 \
  --enable-mixed-chunk \
  --cuda-graph-max-bs 32 \
  --max-running-requests 256
```

---

**文档版本**: v1.0  
**最后更新**: 2025-01-10  
**基于**: SGLang commit `latest`

如有问题或建议，欢迎反馈！🎉

