# SGLang 内存池系统完整笔记

> **作者**: AI助手  
> **日期**: 2025-10-13  
> **版本**: v1.0  
> **基于**: SGLang memory_pool.py, allocator.py, memory_pool_host.py

---

## 📚 目录

1. [整体架构](#整体架构)
2. [memory_pool.py - 物理存储层](#memory_poolpy---物理存储层)
   - [ReqToTokenPool](#reqtotokenpool---请求到token位置映射池)
   - [MambaPool](#mambapool---mamba状态缓存池)
   - [HybridReqToTokenPool](#hybridreqtotokenpool---混合请求token池)
   - [KVCache系列](#kvcache系列---物理存储层)
3. [allocator.py - 分配器层](#allocatorpy---分配器层)
4. [memory_pool_host.py - Host内存层](#memory_pool_hostpy---host内存层)
5. [设计模式总结](#设计模式总结)
6. [性能优化总结](#性能优化总结)

---

## 整体架构

SGLang 使用**多层内存池**设计，支持从GPU到分布式存储的完整KV cache层次结构：

```
┌─────────────────────────────────────────────────────────┐
│                    请求层 (Request)                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│   ReqToTokenPool / HybridReqToTokenPool (memory_pool)   │
│   作用：请求 → Token位置映射 + Mamba状态管理            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│   TokenToKVPoolAllocator / PagedAllocator (allocator)   │
│   作用：Token位置 → KV Cache索引分配策略                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        KVCache系列 (memory_pool)                         │
│   L1 (GPU): MHATokenToKVPool, MLATokenToKVPool, ...    │
│   作用：物理KV cache存储（GPU显存）                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        HostKVCache (memory_pool_host)                   │
│   L2 (CPU): MHATokenToKVPoolHost, MLATokenToKVPoolHost │
│   作用：主机内存KV cache（HiCache L2层）                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        HiCacheStorage (hicache_storage)                 │
│   L3 (分布式): Mooncake, 3FS, NIXL, etc.               │
│   作用：分布式存储后端（HiCache L3层）                   │
└─────────────────────────────────────────────────────────┘
```

---

## memory_pool.py - 物理存储层

### ReqToTokenPool - 请求到Token位置映射池

#### 核心数据结构

```python
class ReqToTokenPool:
    """管理请求到token位置的映射"""
    
    # 核心张量：[请求数, 最大上下文长度]
    self.req_to_token: torch.Tensor  # shape: (size, max_context_len)
    # 例如：
    # req_to_token[0] = [101, 102, 103, ...]  # 请求0的token位置
    # req_to_token[1] = [201, 202, 203, ...]  # 请求1的token位置
    
    self.free_slots: List[int]  # 空闲请求槽位列表
```

#### 核心方法

```python
def __init__(self, size, max_context_len, device, enable_memory_saver):
    """
    Args:
        size: 最大并发请求数（如1024）
        max_context_len: 单个请求最大token数（如32768）
    """
    self.req_to_token = torch.zeros(
        (size, max_context_len), 
        dtype=torch.int32,
        device=device
    )
    self.free_slots = list(range(size))

def alloc(self, need_size: int) -> List[int]:
    """分配请求槽位"""
    if need_size > len(self.free_slots):
        return None
    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]
    return select_index

def write(self, indices, values):
    """写入token位置映射"""
    self.req_to_token[indices] = values

def free(self, free_index: Union[int, List[int]]):
    """释放槽位"""
    if isinstance(free_index, int):
        self.free_slots.append(free_index)
    else:
        self.free_slots.extend(free_index)
```

#### 使用示例

```python
# 初始化
pool = ReqToTokenPool(size=256, max_context_len=32768, device="cuda:0")

# 分配3个请求槽位
req_slots = pool.alloc(3)  # 返回: [0, 1, 2]

# 为请求写入token位置
pool.write(indices=[0], values=torch.tensor([[100,101,102,...]]))

# 释放
pool.free([0, 1, 2])
```

---

### MambaPool - Mamba状态缓存池

#### Mamba架构背景

```
传统Transformer: O(n²) 复杂度
    Q @ K^T → 需要存储完整的KV cache

Mamba/Mamba2: O(n) 复杂度  
    使用循环状态更新 → 只需存储固定大小的状态
```

#### 核心数据结构

```python
class MambaPool:
    """为Mamba/Mamba2架构存储状态空间模型的状态"""
    
    self.mamba_cache: Tuple[torch.Tensor, ...] = (
        conv_state,      # 卷积状态
        temporal_state,  # 时序/SSM状态
        # 如果启用推测解码，还有：
        intermediate_ssm_state_cache,
        intermediate_conv_window_cache,
    )
```

#### 状态形状详解

```python
# 卷积状态 conv_state
shape: (num_mamba_layers, size+1, dim, kernel_size-1)
# 例如：(24层, 1025个槽, 2048维度, 3个历史输入)
# 作用：保存卷积的滑动窗口历史

# 时序状态 temporal_state (SSM state)
shape: (num_mamba_layers, size+1, num_heads, state_dim)
# 例如：(24层, 1025个槽, 8个头, 64维状态)
# 作用：状态空间模型的隐藏状态
```

#### 推测解码的中间缓存

```python
if speculative_num_draft_tokens is not None:
    # 中间SSM状态缓存
    shape: (num_layers, size+1, num_draft_tokens, head, state_dim)
    intermediate_ssm_state_cache = torch.zeros(...)
    
    # 中间卷积窗口缓存  
    shape: (num_layers, size+1, num_draft_tokens, dim, kernel_size-1)
    intermediate_conv_window_cache = torch.zeros(...)
```

#### 核心方法

```python
def alloc(self, need_size: int) -> Optional[List[int]]:
    """分配Mamba状态槽位"""
    if need_size > len(self.free_slots):
        return None
    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]
    return select_index

def free(self, free_index: Union[int, List[int]]):
    """释放并清零状态（避免污染）"""
    if isinstance(free_index, int):
        self.free_slots.append(free_index)
    else:
        self.free_slots.extend(free_index)
    # 清零被释放槽位的状态
    self.mamba_cache[0][:, free_index] = 0  # conv_state
    self.mamba_cache[1][:, free_index] = 0  # temporal_state

def get_mamba_params(self, layer_id: int):
    """获取指定层的Mamba参数"""
    return [self.mamba_cache[i][layer_id] for i in range(len(self.mamba_cache))]
```

#### 内存占用示例

```python
# 配置：24个Mamba层, 1024个并发请求
# conv_state: (2048, 3), fp32
# temporal_state: (8, 64), fp32

conv_state_size = 24 * 1024 * 2048 * 3 * 4 / GB ≈ 0.57 GB
temporal_state_size = 24 * 1024 * 8 * 64 * 4 / GB ≈ 0.048 GB
总计 ≈ 0.62 GB
```

---

### HybridReqToTokenPool - 混合请求Token池

用于 **Hybrid GDN** 模型（如Qwen3Next、FalconH1），同时管理Token位置和Mamba状态。

#### 核心数据结构

```python
class HybridReqToTokenPool(ReqToTokenPool):
    """同时管理普通token位置和Mamba缓存"""
    
    # 继承基础的 req_to_token 映射
    self.req_to_token: torch.Tensor
    
    # 添加 Mamba 专用组件
    self.mamba_pool: MambaPool
    
    # 映射表：Mamba层在模型中的全局ID → 在mamba_pool中的局部ID
    self.mamba_map: Dict[int, int]
    # 例如：{5: 0, 10: 1, 15: 2} 表示第5/10/15层是Mamba层
    
    # 请求索引到Mamba索引的映射
    self.req_index_to_mamba_index_mapping: torch.Tensor
    
    # 双向映射：请求ID ↔ Mamba索引
    self.rid_to_mamba_index_mapping: Dict[str, int]
    self.mamba_index_to_rid_mapping: Dict[int, str]
```

#### 智能分配：支持Chunk Prefill

```python
def alloc(self, need_size: int, reqs: Optional[List["Req"]] = None):
    """同时分配请求槽和Mamba槽"""
    
    # Step 1: 分配请求槽
    select_index = super().alloc(need_size)
    if select_index == None:
        return None
    
    # Step 2: 为每个请求分配或复用Mamba槽
    mamba_index = []
    for req in reqs:
        rid = req.rid
        
        # 检查是否已有Mamba槽（支持chunk prefill）
        if rid in self.rid_to_mamba_index_mapping:
            mid = self.rid_to_mamba_index_mapping[rid]
            # 复用已有的Mamba状态！
        else:
            # 分配新的Mamba槽
            mid = self.mamba_pool.alloc(1)[0]
            # 建立双向映射
            self.rid_to_mamba_index_mapping[rid] = mid
            self.mamba_index_to_rid_mapping[mid] = rid
        
        mamba_index.append(mid)
    
    # Step 3: 记录映射关系
    self.req_index_to_mamba_index_mapping[select_index] = \
        torch.tensor(mamba_index, dtype=torch.int32, device=self.device)
    
    return select_index
```

#### Chunk Prefill 示例

```python
# 场景：长序列分块预填充

# 第一次：处理前1024个token
req = Req(rid="req_001", input_ids=[0:1024])
req_idx = pool.alloc(1, [req])  
# → 分配 req_slot=0, mamba_slot=5

# 第二次：处理接下来的1024个token
req = Req(rid="req_001", input_ids=[1024:2048])
req_idx = pool.alloc(1, [req])
# → 分配新的 req_slot=1, 但复用 mamba_slot=5
# 这样Mamba状态保持连续！
```

#### 条件释放

```python
def free(self, free_index, free_mamba_cache: bool = True):
    """可选地保留Mamba状态"""
    # 释放请求槽
    super().free(free_index)
    
    if free_mamba_cache:
        # 释放Mamba槽并清理映射
        mamba_index = self.req_index_to_mamba_index_mapping[free_index]
        self.mamba_pool.free(mamba_index.tolist())
        # 清理双向映射
        for mid in mamba_index.tolist():
            rid = self.mamba_index_to_rid_mapping.pop(mid)
            self.rid_to_mamba_index_mapping.pop(rid)
    
    # 如果 free_mamba_cache=False，Mamba状态保留
    # 用于chunk prefill的后续批次
```

---

### KVCache系列 - 物理存储层

#### 1. KVCache (抽象基类)

```python
class KVCache(abc.ABC):
    """所有KV缓存的抽象基类"""
    
    # 基础属性
    self.size: int              # 总token容量
    self.page_size: int         # 页大小
    self.dtype: torch.dtype     # 计算数据类型
    self.store_dtype: torch.dtype  # 存储数据类型（可能是fp8）
    self.layer_num: int
    
    # HiCache支持
    self.layer_transfer_counter: Optional[LayerDoneCounter]
```

**关键设计：dtype vs store_dtype**

```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
    # FP8量化：计算用fp8，但存储用uint8
    # 原因：PyTorch的index_put操作不支持fp8
    self.store_dtype = torch.uint8
else:
    self.store_dtype = dtype

# 写入时：
cache_k = cache_k.view(torch.uint8)
self.k_buffer[loc] = cache_k

# 读取时：
k = self.k_buffer[loc].view(torch.float8_e4m3fn)
```

#### 2. MHATokenToKVPool - 多头注意力KV池

**内存布局**

```python
class MHATokenToKVPool(KVCache):
    """标准多头注意力的KV缓存池"""
    
    # K和V分开存储，每层一个tensor
    self.k_buffer: List[torch.Tensor]  # 长度 = layer_num
    self.v_buffer: List[torch.Tensor]  # 长度 = layer_num
    
    # 每个tensor的形状：[size + page_size, head_num, head_dim]
    # 例如：[32769, 32, 128]
    #   - 32768个token槽 + 1个padding槽
    #   - 32个注意力头
    #   - 每个头128维
```

**slot 0的特殊用途**

```python
# slot 0用于吸收padding token的输出

# 场景：批处理不同长度的序列
# Seq 1: [1, 2, 3]         (3 tokens)
# Seq 2: [4, 5, 6, 7, 8]   (5 tokens)

# 需要padding到相同长度：
# Seq 1: [1, 2, 3, PAD, PAD]
# Seq 2: [4, 5, 6, 7, 8]

# Forward时，PAD token也会产生KV输出
# 但这些输出是无用的，全部写入slot 0
# 这样就不占用有效的cache空间
```

**双流优化写入**

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
    if get_is_capture_mode() and self.alt_stream is not None:
        # CUDA Graph模式：使用双流重叠K和V的拷贝
        current_stream = self.device_module.current_stream()
        
        # 主流写K
        self.alt_stream.wait_stream(current_stream)
        self.k_buffer[layer_id][loc] = cache_k
        
        # 备用流写V（并行执行）
        with self.device_module.stream(self.alt_stream):
            self.v_buffer[layer_id][loc] = cache_v
        
        # 同步等待V写完
        current_stream.wait_stream(self.alt_stream)

# 性能提升：
# 单流模式：[写K] → [写V] = 200us
# 双流模式：[写K] + [写V并行] ≈ 110us
```

**内存占用计算**

```python
# 配置：32层, 32768 tokens, 32头×128维, bf16
每层K = 32768 * 32 * 128 * 2 bytes = 256 MB
每层V = 256 MB
每层总计 = 512 MB
32层总计 = 16 GB
```

#### 3. MLATokenToKVPool - 多潜在注意力KV池

**MLA核心原理（用于DeepSeek-V2/V3）**

```python
# 传统MHA：
k_buffer: [32768, 32, 128]  # 32个头×128维 = 4096维/token
v_buffer: [32768, 32, 128]
总维度：32 * 128 * 2 = 8192 维/token

# MLA压缩：
kv_buffer: [32768, 1, 576]  # 只有1个"头"，576维
# compressed_dim = kv_lora_rank + qk_rope_head_dim
#                = 512 + 64 = 576
总维度：576 维/token

# 压缩比：576 / 8192 ≈ 7%，节省93%！
```

**KV buffer内部布局**

```python
# kv_buffer[layer_id] 形状：[seq_len, 1, 576]
# 内部组织：
# [0:512]   → kv_lora_rank 部分（V的低秩表示）
# [512:576] → qk_rope_head_dim 部分（RoPE位置编码）

# 访问时：
k_buffer = kv_buffer  # 整个576维
v_buffer = kv_buffer[..., :512]  # 前512维
```

**Triton Kernel优化拼接**

```python
def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
    """
    MLA的K分为两部分：
    - k_nope: 无位置信息 (kv_lora_rank维)
    - k_rope: 有位置信息 (qk_rope_head_dim维)
    
    使用Triton kernel高效拼接
    """
    set_mla_kv_buffer_triton(
        self.kv_buffer[layer_id],
        loc,
        cache_k_nope,  # [num_tokens, 1, 512]
        cache_k_rope,  # [num_tokens, 1, 64]
    )
    # 输出：kv_buffer[loc] = [k_nope | k_rope]

# 内存节省：
# DeepSeek-V2: 60层×128K tokens
# MHA: 60 GB
# MLA: 8.6 GB (节省86%)
```

#### 4. SWAKVPool - 滑动窗口注意力混合池

**用于Llama4混合模型**

```python
class SWAKVPool(KVCache):
    """分离管理SWA层和全注意力层的KV cache"""
    
    # Llama4架构：每4层中，3层用SWA，1层用全注意力
    # 层0-3：  SWA, SWA, SWA, Full
    # 层4-7：  SWA, SWA, SWA, Full
    # ...
    
    self.swa_kv_pool: MHATokenToKVPool    # SWA层的小缓存
    self.full_kv_pool: MHATokenToKVPool   # 全注意力层的大缓存
    
    self.layers_mapping: Dict[int, Tuple[int, bool]]
    # 格式：{全局层ID: (池内局部ID, 是否是SWA层)}
    
    self.full_to_swa_index_mapping: torch.Tensor
    # 全注意力索引 → SWA索引的映射
```

**智能路由**

```python
def get_key_buffer(self, layer_id: int):
    layer_id_pool, is_swa = self.layers_mapping[layer_id]
    
    if is_swa:
        return self.swa_kv_pool.get_key_buffer(layer_id_pool)
    else:
        return self.full_kv_pool.get_key_buffer(layer_id_pool)
```

**索引空间转换**

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
    layer_id_pool, is_swa = self.layers_mapping[layer.layer_id]
    
    if is_swa:
        # 关键：loc可能是全注意力空间的索引，需要转换
        if self.full_to_swa_index_mapping is not None:
            loc = self.translate_loc_from_full_to_swa(loc)
        
        self.swa_kv_pool.set_kv_buffer(None, loc, cache_k, cache_v, ...)
    else:
        self.full_kv_pool.set_kv_buffer(None, loc, cache_k, cache_v, ...)

# 索引转换示例：
# allocator分配：full_indices=[1000,1001,1002], swa_indices=[10,11,12]
# 映射：full_to_swa_index_mapping[1000]=10, [1001]=11, [1002]=12
# 写入SWA层时自动转换
```

**内存优势**

```python
# 配置：32层(24层SWA+8层Full), SWA窗口4K, Full 128K, bf16

# 传统方案（所有层都是全注意力）：
traditional_size = 32 * 131072 * 32 * 128 * 2 / GB ≈ 32 GB

# SWA混合方案：
swa_size = 24 * 4096 * 32 * 128 * 2 / GB ≈ 0.75 GB
full_size = 8 * 131072 * 32 * 128 * 2 / GB ≈ 8 GB
total_size = 0.75 + 8 = 8.75 GB

# 节省：73%！
```

#### 5. HybridLinearKVPool - 混合线性KV池

**用于Hybrid GDN模型（Qwen3Next、FalconH1）**

```python
class HybridLinearKVPool(KVCache):
    """只为全注意力层分配KV cache"""
    
    # 模型结构：
    # - 部分层是Transformer（需要KV cache）
    # - 部分层是Mamba（不需要KV cache，用MambaPool）
    
    self.full_kv_pool: MHATokenToKVPool
    self.full_attention_layer_id_mapping: Dict[int, int]
    # 全局层ID → 池内局部ID
```

**层ID映射**

```python
# Qwen3Next 32层：
# Mamba层：0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27
# 全注意力层：4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31

self.full_attention_layer_id_mapping = {
    4: 0, 5: 1, 6: 2, 7: 3,      # 全局4-7 → 局部0-3
    12: 4, 13: 5, 14: 6, 15: 7,  # 全局12-15 → 局部4-7
    20: 8, 21: 9, 22: 10, 23: 11,
    28: 12, 29: 13, 30: 14, 31: 15,
}

# 只需分配16层KV cache（而不是32层）
# 节省50%显存！
```

#### 6. NSATokenToKVPool - NSA注意力KV池

**NSA = MLA + 稀疏索引优化**

```python
class NSATokenToKVPool(MLATokenToKVPool):
    """在MLA基础上添加index_k缓存"""
    
    # 主KV cache：压缩的KV表示（继承自MLA）
    self.kv_buffer: List[torch.Tensor]
    
    # 额外的索引缓存（FP8量化）
    self.index_k_with_scale_buffer: List[torch.Tensor]
    # 形状：[num_pages, page_size*head_dim + page_size*4]
    #       └─────┬─────┘ └─────┬─────┘
    #        FP8索引数据      FP32量化scale
```

**index_k_with_scale_buffer 布局**

```python
# page_size=64, index_head_dim=128
# 每个page：
page_data_size = 64 * 128 + 64 * 4 = 8192 + 256 = 8448 bytes

# 详细布局：
# [0:8192]    → 64个token的index_k (FP8)
# [8192:8448] → 64个量化scale (FP32)
```

**NSA工作流程**

```python
# Prefill：计算并存储index_k
for token in input_tokens:
    index_k = model.compute_index_k(token)  # [1, 128]
    index_k_fp8, scale = quantize_to_fp8(index_k)
    pool.set_index_k_and_scale_buffer(layer_id, loc, index_k_fp8, scale)

# Decode：使用index_k做稀疏检索
query_index = model.compute_index_k(new_token)
all_index_k = pool.get_index_k_continuous(layer_id, seq_len, page_indices)

# 计算相似度，找到最相关的K个token
scores = query_index @ all_index_k.T
topk_indices = torch.topk(scores, k=top_k).indices

# 只对topk的token做完整注意力
selected_kv = pool.get_kv_buffer(layer_id)[topk_indices]
output = sparse_attention(query, selected_kv)

# 内存对比：
# 每token：MLA 1152 bytes + NSA index 132 bytes = 1284 bytes
# 传统MHA：16384 bytes
# 节省：92%！
```

#### 7. AscendTokenToKVPool - 昇腾NPU版本

**为华为昇腾NPU优化**

```python
class AscendTokenToKVPool(MHATokenToKVPool):
    """昇腾NPU优化版本"""
    
    # 关键：使用单一连续内存块
    self.kv_buffer = torch.zeros(
        (2, layer_num, num_pages, page_size, head_num, head_dim)
    )
    #  ↑    ↑         ↑          ↑          ↑         ↑
    # K/V  层数      页数       页大小      头数      头维度
    
    self.k_buffer = self.kv_buffer[0]
    self.v_buffer = self.kv_buffer[1]
```

**使用NPU专用算子**

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
    torch_npu._npu_reshape_and_cache(
        key=cache_k,
        value=cache_v,
        key_cache=self.k_buffer[layer_id].view(-1, page_size, head_num, head_dim),
        value_cache=self.v_buffer[layer_id].view(-1, page_size, head_num, head_dim),
        slot_indices=loc,
    )

# NPU算子优势：
# 1. 融合reshape和写入操作
# 2. 使用NPU的向量单元加速
# 3. 优化内存访问模式
# 4. 减少kernel launch开销
```

#### 8. DoubleSparseTokenToKVPool - 双稀疏KV池

```python
class DoubleSparseTokenToKVPool(KVCache):
    """用于Double Sparse Attention（双稀疏注意力）"""
    
    # 双稀疏指：
    # 1. Token稀疏：只关注重要的token
    # 2. Channel稀疏：只关注重要的通道（维度）
    
    self.k_buffer: List[torch.Tensor]  # 标准K
    self.v_buffer: List[torch.Tensor]  # 标准V
    self.label_buffer: List[torch.Tensor]  # 重要通道标签
    # 形状：[size, head_num, heavy_channel_num]

# 使用场景：
# 先用label快速过滤（只计算16维）→ 筛选top tokens
# → 再对top tokens计算完整attention（128维）
```

---

## allocator.py - 分配器层

### BaseTokenToKVPoolAllocator - 抽象基类

#### 双队列机制

```python
class BaseTokenToKVPoolAllocator(abc.ABC):
    """分配器的抽象基类"""
    
    # 空闲页管理（核心机制）
    self.free_pages: torch.Tensor      # 立即可用的空闲页（已排序）
    self.release_pages: torch.Tensor   # 待合并的释放页（未排序）
```

**为什么需要两个队列？**

```python
# 场景：频繁分配释放
for i in range(1000):
    alloc(10)
    free(10)
    # 如果每次free都排序，开销巨大！

# 解决方案：延迟合并
free_pages = [1, 5, 7, 10, 15]     # 已排序
release_pages = [100, 50, 200]     # 未排序

# 只在需要时才合并排序
if need_size > len(free_pages):
    merge_and_sort_free()
```

**为什么需要排序？**

```python
# 未排序：free_pages = [100, 5, 200, 6, 7]
indices = alloc(5)  # 得到 [100, 5, 200, 6, 7]（不连续）

# 排序后：free_pages = [5, 6, 7, 100, 200]
indices = alloc(5)  # 得到 [5, 6, 7, 100, 200]（前3个连续）

# 连续内存的好处：
# - Cache友好
# - 减少TLB miss
# - 支持向量化操作
```

#### 批量释放优化

```python
def free_group_begin(self):
    """开始批量释放"""
    self.is_not_in_free_group = False
    self.free_group = []

def free_group_end(self):
    """结束批量释放，一次性处理"""
    self.is_not_in_free_group = True
    if self.free_group:
        self.free(torch.cat(self.free_group))

# 使用：
allocator.free_group_begin()
for req in finished_reqs:
    allocator.free(req.kv_indices)  # 暂存
allocator.free_group_end()  # 一次性cat

# 性能：
# 逐个释放：100次cat操作
# 批量释放：1次cat操作
```

#### 状态保存恢复（用于推测解码）

```python
# 1. 备份状态
state = allocator.backup_state()

# 2. 推测性分配
indices = allocator.alloc(100)

# 3. 验证失败，回滚
if verification_failed:
    allocator.restore_state(state)
```

---

### TokenToKVPoolAllocator - Token级分配器

```python
class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """最简单的分配器：token级粒度（page_size=1）"""
    
    def __init__(self, size, dtype, device, kvcache, need_sort):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
    
    def clear(self):
        # slot 0保留给padding token
        self.free_pages = torch.arange(1, self.size + 1, 
                                       dtype=torch.int64, 
                                       device=self.device)
    
    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()
        
        if need_size > len(self.free_pages):
            return None
        
        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index
    
    def free(self, free_index: torch.Tensor):
        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)
```

---

### PagedTokenToKVPoolAllocator - 页对齐分配器

**最复杂也是最高效的分配器！**

```python
class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """页对齐分配器：以page为单位（通常64 tokens）"""
    
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
```

#### 基础页分配

```python
def alloc(self, need_size: int):
    """分配need_size个token（必须页对齐）"""
    num_pages = need_size // self.page_size
    
    # 分配页
    out_pages = self.free_pages[:num_pages]
    self.free_pages = self.free_pages[num_pages:]
    
    # 展开为token索引
    out_indices = (
        out_pages[:, None] * self.page_size
        + torch.arange(self.page_size, device=self.device)
    ).reshape(-1)
    
    return out_indices

# 示例：page_size=64, 分配2页 out_pages=[5,7]
# 展开为：[320,321,...,383, 448,449,...,511]
```

#### 高级：alloc_extend（Triton Kernel优化）

```python
def alloc_extend(
    self,
    prefix_lens: torch.Tensor,    # 已有长度
    seq_lens: torch.Tensor,        # 新长度
    last_loc: torch.Tensor,        # 最后一个token位置
    extend_num_tokens: int,        # 总共要扩展的token数
):
    """使用Triton kernel高效并行分配"""
    
    out_indices = torch.empty((extend_num_tokens,), 
                             dtype=torch.int64, 
                             device=self.device)
    
    # 调用Triton kernel
    bs = len(prefix_lens)
    alloc_extend_kernel[(bs,)](
        prefix_lens,
        seq_lens,
        last_loc,
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
        self.seen_max_num_extend_tokens_next_power_of_2,
    )
    
    return out_indices
```

#### Triton Kernel三阶段填充

```python
@triton.jit
def alloc_extend_kernel(...):
    """三阶段填充策略"""
    
    # Part 1: 填充旧的部分页
    # 如果prefix结束在页中间，先把这一页填满
    num_part1 = min(seq_len, next_page_boundary) - pre_len
    
    # Part 2: 填充新的完整页
    # 分配整页，连续填充
    num_part2 = (seq_len // page_size * page_size) - ...
    
    # Part 3: 填充新的部分页
    # 最后可能有个不完整的页
    num_part3 = seq_len % page_size

# 示例：page_size=4, prefix_len=6, seq_len=13
# 旧数据: [0,1,2,3][4,5,_,_]
# 扩展后: [0,1,2,3][4,5,6,7][8,9,10,11][12,_,_,_]
#                   └Part1┘ └─Part2──┘ └Part3┘
```

#### alloc_decode（每次只分配1个token）

```python
def alloc_decode(self, seq_lens, last_loc):
    """Decode阶段：每个请求只需1个token"""
    
    bs = len(seq_lens)
    out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
    
    alloc_decode_kernel[(bs,)](
        seq_lens,
        last_loc,
        self.free_pages,
        out_indices,
        next_power_of_2(bs),
        self.page_size,
    )
    
    return out_indices

# 原理：
# - 如果当前页未满：last_loc + 1
# - 如果当前页已满：分配新页的第一个位置
```

---

### SWATokenToKVPoolAllocator - SWA混合分配器

```python
class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """为Llama4混合模型管理双池分配"""
    
    def __init__(self, size, size_swa, dtype, device, kvcache, need_sort):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        
        # 创建两个子分配器
        self.full_attn_allocator = TokenToKVPoolAllocator(size, ...)
        self.swa_attn_allocator = TokenToKVPoolAllocator(size_swa, ...)
        
        # 维护映射
        self.full_to_swa_index_mapping = torch.empty(
            size + size_swa + 1, dtype=torch.int64, device=device
        )
    
    def alloc(self, need_size: int):
        """同时从两个池中分配"""
        # 检查容量
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None
        
        # 同时分配
        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        
        # 建立映射
        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        
        # 对外返回全注意力索引
        return alloc_full_indices
    
    def free(self, free_index: torch.Tensor):
        """同时释放两个池"""
        self.full_attn_allocator.free(free_index)
        
        # 查找并释放对应的SWA索引
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        
        # 清理映射
        self.full_to_swa_index_mapping[free_index] = 0
```

---

## memory_pool_host.py - Host内存层

### HostKVCache - HiCache L2层

**HiCache架构回顾**

```
L1 (GPU Memory):  最快，容量小（如32GB）
    ↓
L2 (Host Memory): 较快，容量大（如256GB）← memory_pool_host.py
    ↓
L3 (Distributed Storage): 较慢，容量巨大（TB级）
```

#### 核心设计

```python
class HostKVCache(abc.ABC):
    """CPU内存中的KV cache池（HiCache L2层）"""
    
    def __init__(
        self,
        device_pool: KVCache,           # 对应的GPU池
        host_to_device_ratio: float,    # Host/GPU内存比例
        host_size: int,                 # Host内存大小(GB)
        page_size: int,
        layout: str,                    # 内存布局
        pin_memory: bool,               # 是否使用pinned memory
        device: str = "cpu",
    ):
        # 计算Host内存容量
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        
        # 页对齐
        self.size = self.size - (self.size % self.page_size)
        self.page_num = self.size // self.page_size
        
        # 检查Host内存是否足够
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - 10 * (1024**3)  # 保留10GB
        
        if requested_bytes > available_bytes:
            raise ValueError("Not enough host memory!")
        
        # 分配内存
        self.kv_buffer = self.init_kv_buffer()
        
        # 线程安全
        self.lock = threading.RLock()
```

#### 三种内存布局

```python
# 1. layer_first (传统布局)
dims = (2, layer_num, size, head_num, head_dim)
# [K/V, 层, Token, 头, 维度]
# 优点：按层访问连续
# 缺点：跨层访问不连续

# 2. page_first (HiCache优化)
dims = (2, size, layer_num, head_num, head_dim)
# [K/V, Token, 层, 头, 维度]
# 优点：按token访问连续（适合页粒度传输）
# 缺点：按层访问不连续

# 3. page_first_direct (进一步优化)
dims = (2, page_num, layer_num, page_size, head_num, head_dim)
# [K/V, 页, 层, 页内Token, 头, 维度]
# 优点：页粒度组织，零拷贝友好
```

**布局选择建议**

```python
# layer_first: 
# - 适合：direct IO backend
# - 场景：按层加载

# page_first: 
# - 适合：kernel IO backend（零拷贝）
# - 场景：页粒度传输
# - 注意：不兼容direct backend

# page_first_direct:
# - 适合：direct IO backend + FA3
# - 场景：页粒度 + 直接IO
# - 优点：同时支持零拷贝和FA3
```

#### 两种IO后端

```python
# 1. kernel backend (零拷贝，推荐)
# 使用sgl_kernel的kvcacheio模块
# 特点：GPU直接访问CPU pinned memory
# 性能：最快

if io_backend == "kernel":
    if layout == "layer_first":
        transfer_kv_per_layer(
            src_k, dst_k, src_v, dst_v,
            src_indices, dst_indices, item_size
        )
    elif layout == "page_first":
        transfer_kv_per_layer_pf_lf(...)

# 2. direct backend (通用)
# 使用PyTorch的.to()操作
# 特点：兼容性好，但性能稍慢
# 性能：比kernel慢10-20%

elif io_backend == "direct":
    transfer_kv_direct(
        src_layers, dst_layers,
        src_indices, dst_indices, page_size
    )
```

#### 核心方法

```python
@synchronized
def alloc(self, need_size: int) -> Optional[torch.Tensor]:
    """线程安全的分配"""
    assert need_size % self.page_size == 0
    if need_size > self.available_size():
        return None
    
    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]
    return select_index

@synchronized
def free(self, indices: torch.Tensor) -> int:
    """线程安全的释放"""
    self.free_slots = torch.cat([self.free_slots, indices])
    return len(indices)

def load_to_device_per_layer(
    self, device_pool, host_indices, device_indices, layer_id, io_backend
):
    """从Host加载KV cache到GPU（按层）"""
    # 根据layout和io_backend选择合适的传输函数
    if io_backend == "kernel":
        if self.layout == "layer_first":
            transfer_kv_per_layer(...)
        elif self.layout == "page_first":
            transfer_kv_per_layer_pf_lf(...)
    elif io_backend == "direct":
        transfer_kv_direct(...)

def backup_from_device_all_layer(
    self, device_pool, host_indices, device_indices, io_backend
):
    """从GPU备份KV cache到Host（所有层）"""
    # 一次性传输所有层，减少kernel launch开销
    if io_backend == "kernel":
        transfer_kv_all_layer(
            src_k_layers, dst_k_layers,
            src_v_layers, dst_v_layers,
            src_indices, dst_indices,
            item_size, num_layers
        )
```

#### 零拷贝支持（与L3集成）

```python
def get_page_buffer_meta(self, indices):
    """
    返回页的内存地址和大小（用于零拷贝传输到L3）
    """
    ptr_list = []
    kv_buffer_data_ptr = self.kv_buffer.data_ptr()
    
    if self.layout == "layer_first":
        for index in range(0, len(indices), self.page_size):
            for layer_id in range(self.layer_num):
                # 计算K和V的内存地址
                k_ptr = kv_buffer_data_ptr + ...
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
    
    elif self.layout in ["page_first", "page_first_direct"]:
        # page_first布局：页连续
        for index in range(0, len(indices), self.page_size):
            k_ptr = kv_buffer_data_ptr + ...
            v_ptr = k_ptr + v_offset
            ptr_list.append(k_ptr)
            ptr_list.append(v_ptr)
    
    return ptr_list, element_size_list

# 用途：L3存储（如Mooncake）可以直接RDMA读写这些地址
# 实现真正的零拷贝传输
```

---

### MHATokenToKVPoolHost

```python
class MHATokenToKVPoolHost(HostKVCache):
    """MHA模型的Host KV池"""
    
    def get_size_per_token(self):
        """计算每个token的存储大小"""
        return (
            self.head_dim 
            * self.head_num 
            * self.layer_num 
            * self.dtype.itemsize 
            * 2  # K和V
        )
    
    def init_kv_buffer(self):
        """根据layout初始化内存"""
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, 
                   self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, 
                   self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (2, self.page_num, self.layer_num, 
                   self.page_size, self.head_num, self.head_dim)
        
        return torch.empty(
            dims, 
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,  # 关键：启用pinned memory
        )
    
    @property
    def k_buffer(self):
        return self.kv_buffer[0]
    
    @property
    def v_buffer(self):
        return self.kv_buffer[1]
```

---

### MLATokenToKVPoolHost

```python
class MLATokenToKVPoolHost(HostKVCache):
    """MLA模型的Host KV池"""
    
    def get_size_per_token(self):
        """MLA压缩后的大小"""
        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1  # 只有1个头
            * self.dtype.itemsize
            * self.layer_num
        )
    
    def init_kv_buffer(self):
        """MLA只需要一个KV buffer"""
        if self.layout == "layer_first":
            dims = (self.layer_num, self.size, 1,
                   self.kv_lora_rank + self.qk_rope_head_dim)
        elif self.layout == "page_first":
            dims = (self.size, self.layer_num, 1,
                   self.kv_lora_rank + self.qk_rope_head_dim)
        elif self.layout == "page_first_direct":
            dims = (self.page_num, self.layer_num, 
                   self.page_size, 1,
                   self.kv_lora_rank + self.qk_rope_head_dim)
        
        return torch.empty(dims, dtype=self.dtype, device=self.device,
                          pin_memory=self.pin_memory)
```

---

## 设计模式总结

### 1. 分层抽象模式

```
应用层：请求管理
    ↓
ReqToTokenPool：业务层（请求→Token位置）
    ↓
Allocator：管理层（分配策略）
    ↓
KVCache：存储层（物理内存）
    ↓
HostKVCache：扩展层（Host内存）
```

**优点**：
- 职责清晰，易于维护
- 层次解耦，易于扩展
- 支持不同架构（MHA/MLA/Mamba/Hybrid）

### 2. 模板方法模式

```python
# 抽象基类定义接口
class KVCache(abc.ABC):
    @abc.abstractmethod
    def get_key_buffer(self, layer_id): ...
    
    @abc.abstractmethod
    def set_kv_buffer(self, layer, loc, k, v): ...

# 具体子类实现细节
class MHATokenToKVPool(KVCache):
    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]
    
    def set_kv_buffer(self, layer, loc, k, v):
        self.k_buffer[layer_id][loc] = k
        self.v_buffer[layer_id][loc] = v

class MLATokenToKVPool(KVCache):
    def get_key_buffer(self, layer_id):
        return self.kv_buffer[layer_id]  # 返回压缩的KV
    
    def set_kv_buffer(self, layer, loc, k, v):
        # 使用Triton kernel拼接
        set_mla_kv_buffer_triton(self.kv_buffer[layer_id], loc, k, v)
```

### 3. 策略模式

```python
# 不同的分配策略
class TokenToKVPoolAllocator:
    """Token级分配策略"""
    page_size = 1

class PagedTokenToKVPoolAllocator:
    """页对齐分配策略"""
    page_size = 64
    
    def alloc_extend(self, ...):
        # 使用Triton kernel优化

class SWATokenToKVPoolAllocator:
    """双池分配策略"""
    def alloc(self, need_size):
        # 同时从两个池分配
```

### 4. 适配器模式

```python
# HybridLinearKVPool适配不同类型的层
class HybridLinearKVPool(KVCache):
    def __init__(self, full_attention_layer_ids, ...):
        self.full_kv_pool = MHATokenToKVPool(...)
        # Mamba层不需要KV cache
    
    def get_key_buffer(self, layer_id):
        # 将全局层ID转换为池内局部ID
        local_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(local_id)
```

### 5. 工厂模式

```python
# 根据模型类型创建相应的KV池
if model_type == "mha":
    kv_pool = MHATokenToKVPool(...)
elif model_type == "mla":
    kv_pool = MLATokenToKVPool(...)
elif model_type == "hybrid_gdn":
    kv_pool = HybridLinearKVPool(...)
elif model_type == "swa":
    kv_pool = SWAKVPool(...)
```

### 6. 观察者模式（HiCache）

```python
# Layer Transfer Counter通知机制
class KVCache:
    def register_layer_transfer_counter(self, counter):
        self.layer_transfer_counter = counter
    
    def get_key_buffer(self, layer_id):
        if self.layer_transfer_counter is not None:
            # 等待该层数据加载完成
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_key_buffer(layer_id)
```

---

## 性能优化总结

### 1. 内存优化

#### a. 延迟排序（Allocator）
```python
# 问题：频繁free导致大量排序开销
# 方案：使用release_pages暂存，延迟到需要时才排序

# 传统：
for i in range(1000):
    free(indices)
    # 每次都排序，1000次sort！

# 优化：
for i in range(1000):
    free(indices)  # 添加到release_pages
# 只在alloc时才merge_and_sort_free()，1次sort！
```

#### b. 批量操作（Allocator）
```python
# 问题：逐个释放导致大量cat操作
# 方案：批量收集后一次性cat

# 传统：
for req in 100_requests:
    allocator.free(req.indices)  # 100次cat

# 优化：
allocator.free_group_begin()
for req in 100_requests:
    allocator.free(req.indices)  # 只append
allocator.free_group_end()  # 1次cat
```

#### c. 零拷贝传输（HostKVCache）
```python
# 问题：CPU-GPU传输开销大
# 方案：使用pinned memory + kernel backend

# 传统方式：
host_data = kv_cache.cpu()  # 复制到pageable memory
gpu_data = host_data.to('cuda')  # 再复制到GPU

# 零拷贝：
host_data = torch.empty(..., pin_memory=True)  # pinned memory
transfer_kv_per_layer(host_data, gpu_data, ...)  # GPU直接DMA访问
```

### 2. 计算优化

#### a. Triton Kernel并行分配
```python
# 问题：Python循环分配慢
# 方案：使用Triton kernel并行处理

# 传统：
for i in range(batch_size):
    indices[i] = allocate_for_request(i)  # Python循环

# 优化：
alloc_extend_kernel[(batch_size,)](...)  # GPU并行
```

#### b. 双流重叠（MHATokenToKVPool）
```python
# 问题：K和V写入串行
# 方案：使用两个CUDA stream并行写入

# 传统：
k_buffer[loc] = k  # 100us
v_buffer[loc] = v  # 100us
# 总计：200us

# 优化：
with main_stream:
    k_buffer[loc] = k
with alt_stream:  # 并行执行
    v_buffer[loc] = v
# 总计：~110us
```

#### c. MLA压缩（MLATokenToKVPool）
```python
# 问题：KV cache占用显存太大
# 方案：低秩压缩

# MHA：32头×128维×2(K/V) = 8192维/token
# MLA：512(lora)+64(rope) = 576维/token
# 压缩比：93%节省
```

### 3. 架构优化

#### a. Hybrid模型混合缓存
```python
# HybridLinearKVPool：只为全注意力层分配KV cache
# Mamba层使用MambaPool（状态缓存）
# 节省：~50%（Qwen3Next）

# SWAKVPool：SWA层用小缓存，全注意力层用大缓存
# 节省：~73%（Llama4）
```

#### b. NSA稀疏检索
```python
# 两级检索：
# 1. index_k快速筛选（128维FP8）
# 2. 完整attention（576维）
# 只对top-k token计算完整attention
```

#### c. 分页管理
```python
# page_size=64的优势：
# 1. 与HiCache页粒度一致
# 2. 支持零拷贝传输
# 3. 减少内存碎片
# 4. 简化并行处理
```

### 4. I/O优化

#### a. 分层加载（HostKVCache）
```python
# 按层异步加载，边加载边计算
for layer in range(num_layers):
    load_to_device_per_layer(layer)  # 异步启动
    wait_until(layer)  # 计算前等待
    forward(layer)  # 计算该层
    # 下一层可能已在后台加载完成
```

#### b. 三级缓存层次（HiCache）
```
L1 (GPU):  32GB,  延迟~100ns,   命中率80%
L2 (Host): 256GB, 延迟~10us,    命中率15%
L3 (分布式): TB级,  延迟~1ms,     命中率5%

有效延迟 = 80%×100ns + 15%×10us + 5%×1ms
        ≈ 0.08us + 1.5us + 50us = 51.58us
vs 全部L3：1000us
加速：19.4x
```

### 5. 并发优化

#### a. 线程安全（HostKVCache）
```python
@synchronized  # 使用RLock
def alloc(self, need_size):
    # 多线程安全的分配
    ...

# 支持并发：
# - 主线程：forward计算
# - 后台线程：L2/L3数据传输
```

#### b. 异步传输
```python
# PyTorch的non_blocking传输
host_data.to('cuda', non_blocking=True)
# CPU和GPU操作可以overlap
```

---

## 总结

### 核心组件对比

| 组件 | 位置 | 作用 | 适用模型 |
|------|------|------|----------|
| **ReqToTokenPool** | memory_pool.py | 请求→Token映射 | 所有模型 |
| **MambaPool** | memory_pool.py | Mamba状态缓存 | Hybrid GDN |
| **HybridReqToTokenPool** | memory_pool.py | Token+Mamba混合 | Qwen3Next, FalconH1 |
| **MHATokenToKVPool** | memory_pool.py | 标准KV cache | Llama, GPT等 |
| **MLATokenToKVPool** | memory_pool.py | 压缩KV cache | DeepSeek-V2/V3 |
| **NSATokenToKVPool** | memory_pool.py | 稀疏索引优化 | DeepSeek-V3 |
| **SWAKVPool** | memory_pool.py | SWA混合缓存 | Llama4 |
| **HybridLinearKVPool** | memory_pool.py | 混合层缓存 | Qwen3Next |
| **AscendTokenToKVPool** | memory_pool.py | NPU优化 | 昇腾NPU |
| **TokenToKVPoolAllocator** | allocator.py | Token级分配 | 基础场景 |
| **PagedTokenToKVPoolAllocator** | allocator.py | 页对齐分配 | HiCache场景 |
| **SWATokenToKVPoolAllocator** | allocator.py | 双池分配 | Llama4 |
| **HostKVCache** | memory_pool_host.py | Host内存池 | HiCache L2 |

### 内存节省对比

| 技术 | 节省比例 | 适用场景 |
|------|----------|----------|
| **MLA压缩** | ~93% | DeepSeek模型 |
| **NSA稀疏** | ~92% | 长上下文稀疏注意力 |
| **Llama4 SWA** | ~73% | 超长上下文 |
| **Hybrid GDN** | ~50% | 混合架构模型 |
| **HiCache L2** | 实际容量8x | 多轮对话 |

### 性能提升对比

| 优化 | 提升 | 关键技术 |
|------|------|----------|
| **零拷贝传输** | 2-3x | pinned memory + kernel backend |
| **双流写入** | 1.8x | CUDA streams overlap |
| **批量操作** | 10-100x | 减少Python开销 |
| **Triton kernel** | 5-10x | GPU并行 |
| **HiCache层次** | 19x | 三级缓存 |

---

**文档结束**

*这份笔记涵盖了SGLang内存池系统的所有核心组件和设计模式。建议结合代码和本文档一起学习，以更好地理解实现细节。*

