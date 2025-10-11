# SGLang Continuous Batching 深度解析

> 基于 sglang 源码的 Continuous Batching 完全指南

---

## 📚 目录

1. [什么是 Continuous Batching？](#1-什么是-continuous-batching)
2. [核心数据结构](#2-核心数据结构)
3. [主事件循环](#3-主事件循环)
4. [Decode 阶段的 Continuous Batching](#4-decode-阶段的-continuous-batching)
5. [Prefill 阶段的批处理](#5-prefill-阶段的批处理)
6. [Prefill vs Decode 的区别](#6-prefill-vs-decode-的区别)
7. [Chunked Prefill 机制](#7-chunked-prefill-机制)
8. [流量控制与过载保护](#8-流量控制与过载保护)
9. [关键配置参数](#9-关键配置参数)
10. [完整流程示例](#10-完整流程示例)

---

## 1. 什么是 Continuous Batching？

### 1.1 定义

**Continuous Batching** 是一种动态批处理技术，专门用于优化 LLM 推理服务的吞吐量。与传统批处理不同：

- ❌ **传统批处理**：等待一批请求全部完成才能处理下一批
- ✅ **Continuous Batching**：当批次中有请求完成时，立即用新请求填充空位

### 1.2 核心思想

```
维护一个 batch_size 大小的"池子"
   ↓
轮询式执行 (每轮生成 1 token)
   ↓
每轮后 check：是否有请求完成？
   ↓
移除完成的请求，立即塞入新请求
   ↓
循环往复，GPU 永不空闲
```

### 1.3 优势

1. **零等待时间**：请求完成立即被新请求替换
2. **最大化 GPU 利用率**：避免空闲等待
3. **降低平均延迟**：新请求不需要等待整个批次完成
4. **提高吞吐量**：更高的并发处理能力

---

## 2. 核心数据结构

### 2.1 等待队列（Prefill Queue）

```python
# python/sglang/srt/managers/scheduler.py:517
self.waiting_queue: List[Req] = []
```

- **类型**：简单的 Python List
- **作用**：存储所有等待 prefill 的请求
- **排序**：可根据调度策略排序（FCFS、LPM、DFS-Weight 等）

### 2.2 运行批次（Running Batch）

```python
# python/sglang/srt/managers/scheduler.py:519
self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
```

- **类型**：`ScheduleBatch` 对象
- **作用**：当前正在 decode 的请求池
- **动态调整**：每轮 decode 后过滤完成的请求

### 2.3 请求对象（Req）

```python
# python/sglang/srt/managers/schedule_batch.py:434
class Req:
    rid: str                    # 请求 ID
    origin_input_ids: List[int] # 原始输入 tokens
    output_ids: List[int]       # 已生成的 tokens
    sampling_params: SamplingParams
    finished_reason: Optional[str]  # 完成原因
```

---

## 3. 主事件循环

### 3.1 核心循环代码

```python
# python/sglang/srt/managers/scheduler.py:979-995
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:  # ← 无限循环！
        # 1️⃣ 接收新的并发请求
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)  # 加入 waiting_queue
        
        # 2️⃣ 获取下一个要运行的 batch
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch
        
        if batch:
            # 3️⃣ 运行一轮（prefill 或 decode）
            result = self.run_batch(batch)
            
            # 4️⃣ 处理结果：check 是否完成，决定是否塞新 req
            self.process_batch_result(batch, result)
        else:
            # 空闲时自检
            self.self_check_during_idle()
        
        self.last_batch = batch
```

### 3.2 循环流程图

```
┌─────────────────────────────────────────────────────────┐
│                   Event Loop (无限循环)                  │
└─────────────────────────────────────────────────────────┘
     │
     ├─→ recv_requests() ─→ waiting_queue
     │
     ├─→ get_next_batch_to_run()
     │      ├─→ 优先 prefill (new_batch)
     │      └─→ 否则 decode (running_batch)
     │
     ├─→ run_batch() ─→ GPU 前向传播
     │
     ├─→ process_batch_result()
     │      ├─→ filter_batch() (移除完成的请求)
     │      └─→ merge 新 prefill 请求
     │
     └─→ 回到循环开始
```

---

## 4. Decode 阶段的 Continuous Batching

### 4.1 动态更新批次

```python
# python/sglang/srt/managers/scheduler.py:2016-2060
def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
    """Update the current running decoding batch."""
    initial_bs = batch.batch_size()
    
    # 🔥 关键：移除所有已完成的请求
    batch.filter_batch()
    if batch.is_empty():
        batch.batch_is_full = False
        return batch
    
    # 检查内存、处理 retract 等...
    
    if batch.batch_size() < initial_bs:
        batch.batch_is_full = False  # 🔥 批次有空位了！
    
    # 更新批次张量，准备下一轮 decode
    batch.prepare_for_decode()
    return batch
```

### 4.2 过滤已完成请求

```python
# python/sglang/srt/managers/schedule_batch.py:1720-1757
def filter_batch(
    self,
    chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
    keep_indices: Optional[List[int]] = None,
):
    if keep_indices is None:
        # 🔥 核心逻辑：只保留未完成的请求
        keep_indices = [
            i
            for i in range(len(self.reqs))
            if not self.reqs[i].finished()  # ← 关键判断
            and self.reqs[i] not in chunked_req_to_exclude
        ]
    
    # 根据 keep_indices 过滤所有张量
    self.reqs = [self.reqs[i] for i in keep_indices]
    self.req_pool_indices = self.req_pool_indices[keep_indices_device]
    self.seq_lens = self.seq_lens[keep_indices_device]
    # ... 过滤其他张量 ...
```

### 4.3 检查完成条件

```python
# python/sglang/srt/managers/scheduler_output_processor_mixin.py:200-259
def process_batch_result_decode(
    self: Scheduler,
    batch: ScheduleBatch,
    result: GenerationBatchResult,
):
    # 遍历批次中的每个请求
    for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
        if req.is_retracted:
            continue
        
        # 添加生成的 token
        req.output_ids.append(next_token_id)
        
        # 🔥 检查是否完成
        req.check_finished()
        if req.finished():
            # 将 KV cache 缓存起来以便复用
            self.tree_cache.cache_finished_req(req)
            req.time_stats.completion_time = time.perf_counter()
```

### 4.4 Decode 流程示例

```
【初始状态】
waiting_queue: [Req4, Req5]
running_batch: [Req1(token 1/50), Req2(token 1/10), Req3(token 1/100)]
batch_size = 3 (池子大小)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【第 1-9 轮 Decode】
每轮：run_batch() → GPU 并行生成 1 token
结果：所有都未完成

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【第 10 轮 Decode】
run_batch() → GPU 并行生成：
  - Req1: 生成 token 11
  - Req2: 生成 token 11 [EOS] ← 完成了！
  - Req3: 生成 token 11

check: filter_batch()
  → Req2.finished() == True
  → 移除 Req2 ✅

running_batch: [Req1(11/50), Req3(11/100)]  ← 池子有空位了！
batch_is_full = False

get_new_batch_prefill():
  → 从 waiting_queue 取出 Req4
  → 做 prefill

merge_batch():
  → running_batch += Req4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【第 11 轮 Decode】
running_batch: [Req1(11/50), Req3(11/100), Req4(1/20)]  ← 塞进来了！
batch_size = 3 (池子又满了)

run_batch(MIXED) → 继续轮询...
```

---

## 5. Prefill 阶段的批处理

### 5.1 获取 Prefill Batch

```python
# python/sglang/srt/managers/scheduler.py:1867-1951
def get_new_batch_prefill(self):
    # 1️⃣ 对 waiting_queue 排序（根据调度策略）
    self.policy.calc_priority(self.waiting_queue)
    
    # 2️⃣ 创建 PrefillAdder 来管理资源预算
    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        self.running_batch,
        self.new_token_ratio,
        self.max_prefill_tokens,  # ← 限制：最大 prefill tokens
        self.chunked_prefill_size,
        running_bs if self.is_mixed_chunk else 0,
    )
    
    # 3️⃣ 贪心地从 waiting_queue 中取出请求
    for req in self.waiting_queue:
        # 各种检查：LoRA、batch_size、内存...
        
        # 尝试添加这个请求
        res = adder.add_one_req(req, ...)
        
        if res != AddReqResult.CONTINUE:
            # 🔥 第一次失败就停止添加
            break
    
    # 4️⃣ 获取所有能运行的请求
    can_run_list = adder.can_run_list  # 可能有多个！
    
    # 5️⃣ 从 waiting_queue 中移除这些请求
    self.waiting_queue = [
        x for x in self.waiting_queue 
        if x not in set(can_run_list)
    ]
    
    # 6️⃣ 创建一个 batch，包含多个 prefill 请求
    new_batch = ScheduleBatch.init_new(can_run_list, ...)
    
    return new_batch
```

### 5.2 准备 Prefill Batch

```python
# python/sglang/srt/managers/schedule_batch.py:1227-1252
def prepare_for_extend(self):
    self.forward_mode = ForwardMode.EXTEND
    
    # 🔥 关键：提取每个请求的全部输入 tokens
    input_ids = [r.fill_ids[len(r.prefix_indices):] for r in reqs]
    
    # 🔥 计算所有 tokens 的总和
    extend_num_tokens = sum(len(ids) for ids in input_ids)
    
    seq_lens = [len(r.fill_ids) for r in reqs]
    prefix_lens = [len(r.prefix_indices) for r in reqs]
    extend_lens = [r.extend_input_len for r in reqs]
    
    # 🔥 将所有 tokens 拼接成一个大 tensor
    input_ids_tensor = torch.tensor(
        list(chain.from_iterable(input_ids)), dtype=torch.int64
    ).to(self.device, non_blocking=True)
```

### 5.3 执行 Prefill

```python
# python/sglang/srt/model_executor/model_runner.py:1914-1935
def forward_extend(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
    pp_proxy_tensors=None,
) -> LogitsProcessorOutput:
    if not skip_attn_backend_init:
        self.attn_backend.init_forward_metadata(forward_batch)
    
    # 🔥 一次性调用 model.forward，处理所有 tokens
    return self.model.forward(
        forward_batch.input_ids,  # 包含所有请求的所有 tokens
        forward_batch.positions,
        forward_batch,
        **kwargs,
    )
```

### 5.4 Prefill 示例

```
【初始状态】
waiting_queue = [
    Req1 (input: 100 tokens),
    Req2 (input: 200 tokens),
    Req3 (input: 150 tokens),
    Req4 (input: 300 tokens),
    Req5 (input: 50 tokens),
]

假设限制：
- max_prefill_tokens = 500
- max_running_requests = 32

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【调度过程】

1️⃣ calc_priority(waiting_queue) → 排序

2️⃣ PrefillAdder 贪心添加：
   
   尝试 Req1 (100 tokens):
   ✅ 累计 100 tokens ≤ 500
   → can_run_list = [Req1]
   
   尝试 Req2 (200 tokens):
   ✅ 累计 300 tokens ≤ 500
   → can_run_list = [Req1, Req2]
   
   尝试 Req3 (150 tokens):
   ✅ 累计 450 tokens ≤ 500
   → can_run_list = [Req1, Req2, Req3]
   
   尝试 Req4 (300 tokens):
   ❌ 累计 750 tokens > 500  ← 超限！
   → STOP (第一次失败就停止)
   
3️⃣ 创建 batch：
   new_batch = ScheduleBatch([Req1, Req2, Req3])
   forward_mode = EXTEND
   
4️⃣ 更新队列：
   waiting_queue = [Req4, Req5]  ← 剩下的继续等待

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【GPU 执行】

run_batch(new_batch) → 一次性并行处理 3 个 prefill！

注意：某些 attention backend 会串行处理每个序列：
┌─────────────────────────────────────┐
│  for seq_idx in range(3):          │
│    if seq_idx == 0:                │
│      process Req1's 100 tokens     │ ← 一口气
│    if seq_idx == 1:                │
│      process Req2's 200 tokens     │ ← 一口气
│    if seq_idx == 2:                │
│      process Req3's 150 tokens     │ ← 一口气
└─────────────────────────────────────┘

总时间 ≈ max(100, 200, 150) + overhead
🔥 最长的 Req2 制约了总时间！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Prefill 完成后】

process_batch_result():
  → Req1, Req2, Req3 的输入都处理完毕
  → 合并到 running_batch 进入 decode 阶段

running_batch = [Req1(decode), Req2(decode), Req3(decode)]
```

---

## 6. Prefill vs Decode 的区别

### 6.1 核心差异对比

| 特性 | Prefill 阶段 | Decode 阶段 |
|-----|-------------|------------|
| **处理方式** | 一口气处理整个输入 | 每次生成 1 个 token |
| **每次处理 tokens** | 几十到几千 | **1 个** |
| **执行次数** | 1 次（除非 chunked） | 多次（max_new_tokens） |
| **持续时间** | 短（几百毫秒） | **长（几秒到几十秒）** |
| **完成时间差异** | 小（类似输入长度） | **大（不同 max_new_tokens）** |
| **动态调整频率** | 低（一次性完成） | **高（每次迭代）** |
| **Continuous Batching 收益** | 中等 | **非常高** |
| **是否是瓶颈** | 否 | **是** |
| **批处理策略** | 批量打包，并行处理 | 动态增删，轮询执行 |

### 6.2 为什么 Decode 更需要 Continuous Batching？

#### 原因 1：逐 Token 生成，持续时间长

```python
# python/sglang/srt/managers/schedule_batch.py:1463-1464
for req in running_batch.reqs:
    req.extend_input_len = 1  # 🔥 decode 每次只处理 1 个 token
```

#### 原因 2：完成时间不同步

```python
# python/sglang/srt/managers/schedule_batch.py:735-783
def check_finished(self):
    # 检查多种完成条件
    if len(self.output_ids) >= self.sampling_params.max_new_tokens:
        self.finished_reason = FINISH_LENGTH(...)
        return
    
    if last_token_id in stop_token_ids:
        self.finished_reason = FINISH_MATCHED_TOKEN(...)
        return
```

每个请求的 `max_new_tokens` 不同，EOS token 出现时间不同，导致完成时间参差不齐。

#### 原因 3：计算量小，容易有空位

Decode 每次只处理 1 token，计算量小，更容易在批次中腾出空间加入新请求。

---

## 7. Chunked Prefill 机制

### 7.1 触发条件

**Chunked Prefill** 和 **max_prefill_tokens** 是**两个独立的机制**！

```python
# python/sglang/srt/managers/schedule_policy.py:593-637
def add_one_req(self, req: Req, ...):
    # 判断：是否启用 chunked prefill
    if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
        # ✅ 不需要 chunked prefill（正常处理）
        self.can_run_list.append(req)
        self._update_prefill_budget(...)
    else:
        # ❌ 需要 chunked prefill（单个请求输入太长）
        trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
        if trunc_len <= 0:
            return AddReqResult.OTHER
        
        # 🔥 截断输入，分块处理
        req.extend_input_len = trunc_len
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
        
        self.can_run_list.append(req)
        self.new_chunked_req = req  # 标记为 chunked 请求
        self._update_prefill_budget(prefix_len, trunc_len, 0)
```

### 7.2 两个参数的区别

| 参数 | 作用范围 | 限制内容 | 超限后的行为 |
|-----|---------|---------|------------|
| **max_prefill_tokens** | 整个 batch | 所有请求的 tokens 总和 | 停止打包，剩余请求下次处理 |
| **chunked_prefill_size** | 单个请求 | 单个请求的 tokens 数量 | 分块处理该请求（多轮 prefill） |

### 7.3 场景对比

#### 场景 A：超过 max_prefill_tokens

```
waiting_queue = [Req1(100), Req2(200), Req3(300)]
max_prefill_tokens = 400
chunked_prefill_size = None (禁用)

打包：
  ✅ Req1 (100) → 累计 100
  ✅ Req2 (200) → 累计 300
  ❌ Req3 (300) → 累计 600 > 400 ← 超限！
  → break

结果：这次处理 [Req1, Req2]，Req3 下次再处理
→ 不是 chunked prefill，只是延后处理
```

#### 场景 B：触发 chunked_prefill

```
waiting_queue = [Req1(5000 tokens)]
max_prefill_tokens = 8192
chunked_prefill_size = 2048  ← 启用！

打包：
  尝试 Req1 (5000):
    5000 > 2048 ← 触发 chunked prefill！
    
    第一轮：处理 tokens [0, 2048)
    第二轮：处理 tokens [2048, 4096)
    第三轮：处理 tokens [4096, 5000)

结果：一个大请求被拆成 3 次 prefill
```

---

## 8. 流量控制与过载保护

### 8.1 请求流程

```
┌──────────────────┐
│  Client (HTTP)   │
└────────┬─────────┘
         │ 1️⃣ TCP 连接
         ↓
┌─────────────────────────────┐
│  HTTP Server                │
│  (FastAPI/uvicorn)          │
│  - max_connections 限制     │
└────────┬────────────────────┘
         │ 2️⃣ 异步事件循环
         ↓
┌─────────────────────────────┐
│  TokenizerManager           │
│  (asyncio event loop)       │
│  - 异步队列                 │
│  - 不会无限增长             │
└────────┬────────────────────┘
         │ 3️⃣ ZMQ pipe
         ↓
┌─────────────────────────────┐
│  Scheduler                  │
│  - waiting_queue: List[Req] │ ← 🔥 核心阻塞点
│  - max_queued_requests 限制 │
└─────────────────────────────┘
```

### 8.2 队列溢出保护

```python
# python/sglang/srt/managers/scheduler.py:1488-1568
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if self.disaggregation_mode == DisaggregationMode.NULL:
        self._set_or_validate_priority(req)
        
        # 🔥 关键：检查队列是否满
        if self._abort_on_queued_limit(req):
            return  # 拒绝这个请求
        
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)

def _abort_on_queued_limit(self, recv_req: Req) -> bool:
    """Abort an incoming or existing request if the waiting queue is full."""
    
    # 检查队列大小
    if (
        self.max_queued_requests is None
        or len(self.waiting_queue) + 1 <= self.max_queued_requests
    ):
        return False  # 队列还有空间
    
    # 🔥 队列满了！
    req_to_abort = recv_req
    message = "The request queue is full."
    
    if self.enable_priority_scheduling:
        # 🔥 优先级调度：可能踢掉低优先级的旧请求
        direction = 1 if self.schedule_low_priority_values_first else -1
        key_fn = lambda item: (
            direction * item[1].priority,
            item[1].time_stats.wait_queue_entry_time,
        )
        idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
        abort_existing_req = (
            direction * recv_req.priority < direction * candidate_req.priority
        )
        if abort_existing_req:
            self.waiting_queue.pop(idx)  # 踢掉旧请求
            req_to_abort = candidate_req
            message = "The request is aborted by a higher priority request."
    
    # 🔥 发送 abort 消息
    self.send_to_tokenizer.send_pyobj(
        AbortReq(
            finished_reason={
                "type": "abort",
                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,  # 503
                "message": message,
            },
            rid=req_to_abort.rid,
        )
    )
    return req_to_abort.rid == recv_req.rid
```

### 8.3 三层防护机制

| 层级 | 组件 | 保护机制 | 超限行为 |
|-----|------|---------|---------|
| **Level 1** | HTTP Server | `max_connections` | 拒绝 TCP 连接 |
| **Level 2** | TokenizerManager | asyncio 异步队列 | 自然背压 |
| **Level 3** | Scheduler | `max_queued_requests` | 返回 503 错误 |

### 8.4 实际表现

```python
# 配置示例
server_args = {
    "max_queued_requests": 1000,
    "max_prefill_tokens": 8192,
    "chunked_prefill_size": 4096,
}

# 场景：疯狂发请求
for i in range(10000):
    requests.post("/v1/completions", ...)

# 结果：
#   - 前 1000 个：进入 waiting_queue
#   - 后面的：收到 503 错误
#     {
#       "error": {
#         "type": "abort",
#         "status_code": 503,
#         "message": "The request queue is full."
#       }
#     }
#
# 服务器状态：
#   ✅ 内存可控（最多 1000 个请求在等待）
#   ✅ 不会 OOM
#   ✅ 继续处理队列中的请求
#   ✅ 新请求被拒绝，但服务不崩溃
```

---

## 9. 关键配置参数

### 9.1 参数说明

| 参数 | 类型 | 默认值 | 作用 |
|-----|------|--------|------|
| `max_total_num_tokens` | int | 自动计算 | KV cache 总容量 |
| `max_prefill_tokens` | int | 自动计算 | 一次 prefill batch 的最大 tokens 数 |
| `max_running_requests` | int | 自动计算 | running_batch 的最大请求数 |
| `max_queued_requests` | int | None | waiting_queue 的最大请求数（None = 无限制） |
| `chunked_prefill_size` | int | None | 单个请求的最大 tokens 数（None = 禁用） |
| `schedule_policy` | str | "lpm" | 调度策略（fcfs/lpm/dfs-weight/lof/random） |

### 9.2 配置示例

```python
from sglang import ServerArgs

# 生产环境推荐配置
server_args = ServerArgs(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    
    # 流量控制
    max_queued_requests=2000,      # 最多排队 2000 个请求
    
    # 批处理配置
    max_prefill_tokens=16384,      # 一次 prefill 最多 16K tokens
    chunked_prefill_size=8192,     # 单个请求超过 8K 就分块
    max_running_requests=256,      # decode 池子最多 256 个请求
    
    # 调度策略
    schedule_policy="lpm",         # 最长前缀匹配（利用 KV cache）
    enable_priority_scheduling=True,  # 启用优先级调度
    
    # 内存管理
    mem_fraction_static=0.9,       # 90% GPU 内存用于 KV cache
)
```

---

## 10. 完整流程示例

### 10.1 时间线示例

```
时刻 T0: 系统初始化
  waiting_queue = []
  running_batch = []

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T1: 3 个请求到达
  HTTP POST → TokenizerManager → Scheduler
  waiting_queue = [Req1(100 tokens), Req2(200 tokens), Req3(150 tokens)]
  running_batch = []

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T2: 第 1 轮循环 - Prefill
  get_next_batch_to_run():
    → get_new_batch_prefill()
    → 打包 [Req1, Req2, Req3]
    → new_batch.prepare_for_extend()
  
  run_batch(new_batch):
    → ForwardMode.EXTEND
    → GPU 一次性处理 450 tokens
    → 所有请求 prefill 完成
  
  process_batch_result():
    → 合并到 running_batch
  
  waiting_queue = []
  running_batch = [Req1(decode), Req2(decode), Req3(decode)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T3: 第 2 轮循环 - Decode
  get_next_batch_to_run():
    → get_new_batch_prefill() = None (队列空)
    → update_running_batch(running_batch)
    → running_batch.prepare_for_decode()
  
  run_batch(running_batch):
    → ForwardMode.DECODE
    → GPU 生成 3 个 tokens (每个 req 1 个)
  
  process_batch_result():
    → filter_batch() → 所有都未完成
  
  running_batch = [Req1(1/50), Req2(1/10), Req3(1/100)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T4-T11: 第 3-10 轮循环 - Decode
  持续 decode...
  
  新请求到达：Req4(120 tokens), Req5(80 tokens)
  waiting_queue = [Req4, Req5]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T12: 第 11 轮循环 - Req2 完成
  run_batch(running_batch):
    → Req2 生成 EOS token
  
  process_batch_result():
    → filter_batch()
    → Req2.finished() = True
    → 移除 Req2 ✅
    → batch_is_full = False
  
  running_batch = [Req1(10/50), Req3(10/100)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T13: 第 12 轮循环 - 加入新请求
  get_next_batch_to_run():
    → get_new_batch_prefill()
    → 打包 [Req4, Req5] (200 tokens)
    → new_batch.prepare_for_extend()
    → new_batch.mix_with_running(running_batch) 🔥
  
  run_batch(new_batch):
    → ForwardMode.MIXED  🔥 混合模式！
    → GPU 同时处理：
        - Req4 的 120 tokens (prefill)
        - Req5 的 80 tokens (prefill)
        - Req1 的 1 token (decode)
        - Req3 的 1 token (decode)
  
  process_batch_result():
    → Req4, Req5 prefill 完成，进入 decode
    → 合并到 running_batch
  
  waiting_queue = []
  running_batch = [Req1(11/50), Req3(11/100), Req4(1/20), Req5(1/30)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

时刻 T14-...: 继续轮询 decode
  running_batch 中的请求逐个完成
  新请求到达时立即填充
  
  循环往复，GPU 永不空闲... 🔄
```

### 10.2 Forward Mode 转换图

```
┌─────────────────────────────────────────────────────────┐
│                    Forward Mode 状态机                   │
└─────────────────────────────────────────────────────────┘

新请求到达 (waiting_queue 非空)
    ↓
EXTEND (Prefill)
    ├─→ 所有请求 prefill 完成 → 合并到 running_batch
    └─→ 有运行中的 decode → MIXED
              ↓
        MIXED (混合模式)
            ├─→ prefill + decode 同时处理
            └─→ prefill 完成，全部进入 decode
                      ↓
                DECODE (逐 token 生成)
                    ├─→ 每轮生成 1 token
                    ├─→ filter_batch() 移除完成的
                    └─→ 有新请求 → MIXED
                              ↓
                        循环往复...
```

---

## 📝 总结

### 核心要点

1. **Continuous Batching 的本质**：维护一个动态的请求池，每轮 decode 后立即移除完成的请求并加入新请求。

2. **Decode 是主战场**：由于逐 token 生成、持续时间长、完成时间不同步，decode 阶段是 continuous batching 收益最大的地方。

3. **Prefill 批量打包**：多个新请求会被打包成一个 batch 并行处理，但遇到资源限制会立即停止（第一次失败就停）。

4. **一口气 vs 逐 token**：
   - Prefill：一口气处理每个请求的全部输入 tokens
   - Decode：每次只生成 1 个 token，需要多轮迭代

5. **两个独立机制**：
   - `max_prefill_tokens`：限制一次 batch 的总 tokens，超限则延后
   - `chunked_prefill_size`：限制单个请求的 tokens，超限则分块

6. **完善的过载保护**：三层防护机制确保疯狂发请求也不会崩溃，返回 503 而不是 OOM。

### 性能优化建议

1. **合理设置 `max_prefill_tokens`**：太小导致吞吐量低，太大导致延迟高
2. **启用 `chunked_prefill_size`**：处理超长输入时避免 OOM
3. **配置 `max_queued_requests`**：防止内存无限增长
4. **选择合适的调度策略**：`lpm` 对缓存友好，`fcfs` 公平
5. **启用优先级调度**：保护重要请求

### 代码位置索引

| 功能 | 文件路径 | 行数 |
|-----|---------|------|
| 主事件循环 | `python/sglang/srt/managers/scheduler.py` | 979-995 |
| 更新 running_batch | `python/sglang/srt/managers/scheduler.py` | 2016-2060 |
| 过滤完成请求 | `python/sglang/srt/managers/schedule_batch.py` | 1720-1757 |
| 获取 prefill batch | `python/sglang/srt/managers/scheduler.py` | 1835-2014 |
| PrefillAdder | `python/sglang/srt/managers/schedule_policy.py` | 315-698 |
| 队列溢出保护 | `python/sglang/srt/managers/scheduler.py` | 1528-1568 |
| Forward Mode | `python/sglang/srt/model_executor/forward_batch_info.py` | 62-129 |

---

**文档版本**：基于 sglang 源码分析（2025-01）  
**作者**：AI Assistant  
**最后更新**：2025-01-10

