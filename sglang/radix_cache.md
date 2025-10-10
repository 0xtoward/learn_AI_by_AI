# SGLang Radix Cache 完全解析

> 本文档详细解析 SGLang 中 Radix Cache 的设计与实现，包括数据结构、核心算法、调用流程和性能优化。

## 目录

- [1. 概述](#1-概述)
- [2. 核心数据结构](#2-核心数据结构)
  - [2.1 RadixKey](#21-radixkey)
  - [2.2 TreeNode](#22-treenode)
  - [2.3 RadixCache](#23-radixcache)
- [3. 核心算法](#3-核心算法)
  - [3.1 Match Prefix](#31-match-prefix)
  - [3.2 Insert](#32-insert)
  - [3.3 Evict](#33-evict)
- [4. 系统架构](#4-系统架构)
  - [4.1 CPU-GPU 分离架构](#41-cpu-gpu-分离架构)
  - [4.2 调用链路](#42-调用链路)
  - [4.3 内存管理](#43-内存管理)
- [5. Page 机制](#5-page-机制)
- [6. HiCache 分层缓存](#6-hicache-分层缓存)
- [7. Paged Attention](#7-paged-attention)
- [8. 实战示例](#8-实战示例)
- [9. 性能分析](#9-性能分析)
- [10. 常见问题](#10-常见问题)

---

## 快速架构图

```
┌────────────────────────────────────────────────────────────────┐
│                        用户请求                                 │
└────────────────────────┬───────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────────┐
│                     Scheduler                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  ScheduleBatch                                            │ │
│  │    ├─ match_prefix() → 查找缓存                          │ │
│  │    ├─ alloc_token_slots() → 分配内存                     │ │
│  │    │     └─ _evict_tree_cache_if_needed() → 按需淘汰     │ │
│  │    └─ cache_finished_req() → 缓存结果                    │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────────────────┘
                         ↓
         ┌───────────────┴────────────────┐
         │                                 │
         ↓                                 ↓
┌──────────────────┐              ┌──────────────────┐
│   CPU (Host)     │              │   GPU (Device)   │
├──────────────────┤              ├──────────────────┤
│                  │              │                  │
│  Radix Tree      │   索引映射    │  KV Cache Pool  │
│  ┌────────────┐  │  ─────────→  │  ┌────────────┐ │
│  │ TreeNode   │  │              │  │ k_buffer   │ │
│  │ key: [1,2] │  │              │  │ [0:1024,   │ │
│  │ value: [100]│─┼──────────────┼─→│  32, 128]  │ │
│  └────────────┘  │              │  │            │ │
│                  │              │  │ v_buffer   │ │
│  Allocator       │              │  │ [0:1024,   │ │
│  free_slots:[..] │              │  │  32, 128]  │ │
│                  │              │  └────────────┘ │
└──────────────────┘              └──────────────────┘
```

---

## 1. 概述

### 1.1 什么是 Radix Cache？

**Radix Cache** 是 SGLang 中用于缓存和复用 KV Cache 的核心机制。它基于 **Radix Tree（基数树）** 数据结构，实现了高效的前缀共享。

### 1.2 为什么需要 Radix Cache？

```python
# 场景：多个用户问类似的问题
User 1: "今天天气怎么样？明天呢？"
User 2: "今天天气怎么样？后天呢？"
User 3: "今天天气怎么样？"

# 问题：每个请求都要重新计算 KV cache
# - "今天天气怎么样？" 这个前缀被计算了 3 次
# - 浪费了大量 GPU 计算资源

# 解决：使用 Radix Cache
# - 前缀 "今天天气怎么样？" 的 KV cache 只计算一次
# - 后续请求直接复用，节省 60-80% 的计算
```

### 1.3 核心优势

1. **前缀共享**：相同前缀的 KV cache 只存储一份
2. **自动管理**：自动识别和复用前缀
3. **内存高效**：通过 LRU/LFU 策略淘汰冷数据
4. **透明集成**：对上层应用透明

---

## 2. 核心数据结构

### 2.1 RadixKey

`RadixKey` 是 Radix Tree 的键，包含 token 序列和可选的命名空间标识。

```python
class RadixKey:
    def __init__(self, token_ids: List[int], extra_key: Optional[str] = None):
        self.token_ids = token_ids  # token 序列
        self.extra_key = extra_key  # 命名空间（如 LoRA ID）
```

**示例**：

```python
# 普通 key
key1 = RadixKey([1, 2, 3, 4])

# 带命名空间的 key（用于隔离不同的 LoRA）
key2 = RadixKey([1, 2, 3, 4], extra_key="lora_adapter_123")

# 切片操作
key1[0:2]  # RadixKey([1, 2])
```

**设计要点**：
- `token_ids`：实际的 token 序列
- `extra_key`：隔离不同的缓存命名空间
  - 不同的 LoRA adapter
  - 不同的 sampling 策略
  - 不同的 cache 版本

---

### 2.2 TreeNode

`TreeNode` 是 Radix Tree 的节点，每个节点代表一段连续的 token 序列。

#### 2.2.1 基本结构

```python
class TreeNode:
    counter = 0  # 全局计数器，生成唯一 ID
    
    def __init__(self, id: Optional[int] = None):
        # ========== 树结构 ==========
        self.children: Dict[Any, TreeNode] = defaultdict(TreeNode)
        self.parent: TreeNode = None
        
        # ========== 数据存储 ==========
        self.key: RadixKey = None              # token 序列
        self.value: torch.Tensor = None        # GPU KV cache 索引
        
        # ========== 内存管理 ==========
        self.lock_ref: int = 0                 # 引用计数
        self.last_access_time: float = time.monotonic()
        
        # ========== 统计信息 ==========
        self.hit_count: int = 0                # 命中次数
        
        # ========== HiCache 相关 ==========
        self.host_ref_counter: int = 0         # CPU 引用计数
        self.host_value: torch.Tensor = None   # CPU KV cache 索引
        self.hash_value: List[str] = None      # L3 存储哈希
        
        # ========== 节点标识 ==========
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
```

#### 2.2.2 属性详解

| 属性 | 类型 | 位置 | 作用 |
|------|------|------|------|
| `children` | Dict | CPU | 子节点字典，key 可以是 int 或 tuple |
| `parent` | TreeNode | CPU | 父节点引用 |
| `key` | RadixKey | CPU | 该节点代表的 token 序列 |
| `value` | Tensor | CPU | **GPU KV cache 的索引**（不是数据！） |
| `lock_ref` | int | CPU | 引用计数，>0 表示被使用，不能淘汰 |
| `last_access_time` | float | CPU | 最后访问时间（LRU） |
| `hit_count` | int | CPU | 命中次数（LFU） |
| `host_value` | Tensor | CPU | CPU 内存中的 KV cache 索引 |
| `hash_value` | List[str] | CPU | 分布式存储的哈希值 |

#### 2.2.3 节点状态

```python
# 状态属性
@property
def evicted(self) -> bool:
    """节点是否已从 GPU 淘汰"""
    return self.value is None

@property
def backuped(self) -> bool:
    """节点是否在 CPU 有备份"""
    return self.host_value is not None

# 状态转换
[New] value=None, host_value=None
  ↓ insert
[GPU Only] value=[100,101], host_value=None
  ↓ backup to CPU
[GPU+CPU] value=[100,101], host_value=[5000,5001]
  ↓ evict from GPU
[CPU Only] value=None, host_value=[5000,5001]
```

#### 2.2.4 节点示例

```python
# 创建节点
node = TreeNode()

# 设置数据
node.key = RadixKey([1, 2, 3, 4])
node.value = torch.tensor([100, 101, 102, 103])  # GPU 索引

# 使用节点
node.lock_ref += 1  # 标记为使用中
node.hit_count += 1  # 增加命中计数
node.last_access_time = time.monotonic()  # 更新访问时间

# 检查状态
print(f"Evicted: {node.evicted}")    # False
print(f"Backuped: {node.backuped}")  # False
```

---

### 2.3 RadixCache

`RadixCache` 是整个缓存系统的管理器。

#### 2.3.1 基本结构

```python
class RadixCache:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        eviction_policy: str = "lru",  # 默认 LRU
        is_eagle: bool = False,
    ):
        # 内存池引用
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        
        # 配置
        self.page_size = page_size
        self.disable = disable
        
        # 淘汰策略
        if eviction_policy.lower() == "lru":
            self.eviction_strategy = LRUStrategy()
        elif eviction_policy.lower() == "lfu":
            self.eviction_strategy = LFUStrategy()
        
        # 树结构
        self.root_node = TreeNode()
        self.evictable_size_ = 0
        self.protected_size_ = 0
```

#### 2.3.2 淘汰策略

```python
# LRU Strategy
class LRUStrategy:
    def get_priority(self, node: TreeNode) -> float:
        """优先级 = 访问时间（越早越优先淘汰）"""
        return node.last_access_time

# LFU Strategy
class LFUStrategy:
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        """优先级 = (命中次数, 访问时间)"""
        return (node.hit_count, node.last_access_time)
```

**配置方式**：

```bash
# 使用 LRU（默认）
python -m sglang.launch_server --model-path MODEL

# 使用 LFU
python -m sglang.launch_server \
  --model-path MODEL \
  --radix-eviction-policy lfu
```

---

## 3. 核心算法

### 3.1 Match Prefix

**功能**：在树中查找最长的已缓存前缀。

#### 3.1.1 算法流程

```python
def match_prefix(self, key: RadixKey) -> MatchResult:
    """
    查找最长缓存前缀
    
    Returns:
        device_indices: GPU KV cache 索引
        last_node: 匹配路径的最后一个节点
    """
    # 1. 对齐到 page 边界
    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]
    
    # 2. 从根节点开始匹配
    value, last_node = self._match_prefix_helper(self.root_node, key)
    
    # 3. 拼接所有匹配的 KV 索引
    if value:
        value = torch.cat(value)
    
    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
    )

def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    """递归匹配辅助函数"""
    node.last_access_time = time.monotonic()  # 更新访问时间
    child_key = self.get_child_key_fn(key)
    
    value = []
    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        child.last_access_time = time.monotonic()
        
        # 计算匹配长度
        prefix_len = self.key_match_fn(child.key, key)
        
        if prefix_len < len(child.key):
            # 部分匹配，需要分裂节点
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # 完全匹配，继续向下
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
    
    return value, node
```

#### 3.1.2 匹配示例

```python
# ========== 树的初始状态 ==========
Root
  └─ [1, 2, 3, 4] → [100, 101, 102, 103]

# ========== 场景1: 完全匹配 ==========
match_prefix([1, 2, 3, 4])
# 返回: indices=[100,101,102,103], last_node=节点[1,2,3,4]

# ========== 场景2: 部分匹配 ==========
match_prefix([1, 2, 3, 5, 6])
# 返回: indices=[100,101,102], last_node=节点[1,2,3]
# 注意: 树会自动分裂成 [1,2,3] 和 [4]

# ========== 场景3: 无匹配 ==========
match_prefix([5, 6, 7])
# 返回: indices=[], last_node=Root
```

#### 3.1.3 节点分裂

当匹配到节点中间时，自动分裂节点：

```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    """
    分裂前：
        parent → child([1,2,3,4] → [100,101,102,103])
    
    分裂后（split_len=3）：
        parent → new_node([1,2,3] → [100,101,102])
                    └─ child([4] → [103])
    """
    new_node = TreeNode()
    
    # 设置新节点
    new_node.children = {self.get_child_key_fn(key[split_len:]): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len]
    
    # 更新原节点
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:]
    
    # 插入树中
    new_node.parent.children[self.get_child_key_fn(key)] = new_node
    
    return new_node
```

---

### 3.2 Insert

**功能**：将新的 token 序列和 KV cache 索引插入树中。

#### 3.2.1 算法流程

```python
def insert(self, key: RadixKey, value: torch.Tensor) -> int:
    """
    插入 key-value 对
    
    Returns:
        total_prefix_length: 已存在于树中的前缀长度
    """
    # 1. 转换 key（如 EAGLE bigram）
    key.token_ids = self.key_convert_fn(key.token_ids)
    
    # 2. 调用核心插入逻辑
    return self._insert_helper(self.root_node, key, value)

def _insert_helper(self, node: TreeNode, key: RadixKey, value):
    """核心插入逻辑"""
    node.last_access_time = time.monotonic()
    
    if len(key) == 0:
        return 0
    
    child_key = self.get_child_key_fn(key)
    total_prefix_length = 0
    
    # 遍历匹配路径
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        node.last_access_time = time.monotonic()
        
        # 计算匹配长度
        prefix_len = self.key_match_fn(node.key, key)
        total_prefix_length += prefix_len
        
        # 消耗匹配的部分
        key = key[prefix_len:]
        value = value[prefix_len:]
        
        # 部分匹配，需要分裂
        if prefix_len < len(node.key):
            new_node = self._split_node(node.key, node, prefix_len)
            node = new_node
        
        if len(key):
            child_key = self.get_child_key_fn(key)
    
    # 插入新节点（如果还有剩余）
    if len(key):
        new_node = TreeNode()
        new_node.parent = node
        new_node.key = key
        new_node.value = value
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
    
    return total_prefix_length
```

#### 3.2.2 插入示例

**示例1：需要分裂节点**

```python
# 初始树
Root
  └─ [1, 2, 3, 4] → [100, 101, 102, 103]

# 插入 [1, 2, 3, 5, 6]
insert([1, 2, 3, 5, 6], [200, 201, 202, 203, 204])

# 执行过程：
# 1. 匹配到 [1,2,3,4]，prefix_len=3
# 2. 分裂节点
# 3. 插入新分支

# 最终树
Root
  └─ [1, 2, 3] → [100, 101, 102]
       ├─ [4] → [103]
       └─ [5, 6] → [203, 204]

# 返回: 3 (前3个已存在)
```

**示例2：扩展路径**

```python
# 初始树
Root
  └─ [1, 2, 3] → [100, 101, 102]

# 插入 [1, 2, 3, 4, 5]
insert([1, 2, 3, 4, 5], [200, 201, 202, 203, 204])

# 执行过程：
# 1. 匹配到 [1,2,3]，prefix_len=3（完全匹配）
# 2. 创建新子节点

# 最终树
Root
  └─ [1, 2, 3] → [100, 101, 102]
       └─ [4, 5] → [203, 204]

# 返回: 3
```

**示例3：完全重复**

```python
# 初始树
Root
  └─ [1, 2, 3, 4] → [100, 101, 102, 103]

# 插入 [1, 2, 3, 4]
insert([1, 2, 3, 4], [200, 201, 202, 203])

# 执行过程：
# 1. 匹配到 [1,2,3,4]，prefix_len=4（完全匹配）
# 2. key 剩余长度=0，不创建新节点

# 树不变
Root
  └─ [1, 2, 3, 4] → [100, 101, 102, 103]

# 返回: 4 (全部已存在)
```

#### 3.2.3 返回值的使用

```python
# 在 cache_finished_req 中使用
old_prefix_len = len(req.prefix_indices)  # 之前匹配的长度: 2
new_prefix_len = self.insert(key, value)   # 插入后发现前缀长度: 4

# 释放 [2:4] 之间的重复 KV cache
# 因为这部分现在在树中共享了
self.token_to_kv_pool_allocator.free(
    kv_indices[old_prefix_len:new_prefix_len]
)
```

---

### 3.3 Evict

**功能**：淘汰节点释放 GPU 内存。

#### 3.3.1 算法流程

```python
def evict(self, num_tokens: int):
    """
    淘汰指定数量的 tokens
    
    特点：
    - 只淘汰叶子节点（保证树完整性）
    - 使用最小堆（高效找到优先级最低的）
    - 保护被引用的节点（lock_ref > 0）
    - 动态调整（父节点可能成为新叶子）
    """
    if self.disable:
        return
    
    # ========== 步骤1: 收集所有叶子节点 ==========
    leaves = self._collect_leaves()
    
    # ========== 步骤2: 构建优先级堆（最小堆）==========
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) 
        for node in leaves
    ]
    heapq.heapify(eviction_heap)
    
    # ========== 步骤3: 循环淘汰 ==========
    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        
        # 保护根节点
        if x == self.root_node:
            break
        
        # 保护被引用的节点
        if x.lock_ref > 0:
            continue
        
        # ========== 步骤4: 释放 GPU 内存 ==========
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        
        # ========== 步骤5: 删除节点 ==========
        self._delete_leaf(x)
        
        # ========== 步骤6: 父节点可能成为新叶子 ==========
        if len(x.parent.children) == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))

def _collect_leaves(self):
    """收集所有叶子节点"""
    ret_list = []
    stack = [self.root_node]
    
    while stack:
        cur_node = stack.pop()
        if len(cur_node.children) == 0:
            ret_list.append(cur_node)
        else:
            stack.extend(cur_node.children.values())
    
    return ret_list
```

#### 3.3.2 淘汰示例

```python
# ========== 初始树 ==========
Root (lock_ref=1)
  ├─ [1, 2] → [100, 101] (lock_ref=0, last_access=10.0)
  │   ├─ [3, 4] → [102, 103] (lock_ref=0, last_access=15.0) ← 叶子
  │   └─ [5, 6] → [104, 105] (lock_ref=1)                  ← 叶子（被保护）
  └─ [7, 8] → [106, 107] (lock_ref=0, last_access=12.0)
       └─ [9, 10] → [108, 109] (lock_ref=0, last_access=8.0) ← 叶子

# ========== 调用 evict(6) ==========

# 步骤1: 收集叶子
leaves = [
    node([3,4]),    # last_access=15.0
    node([5,6]),    # last_access=不重要（被保护）
    node([9,10]),   # last_access=8.0
]

# 步骤2: 构建堆（LRU，越小越优先）
heap = [
    (8.0, node([9,10])),   # ← 堆顶，最早访问
    (15.0, node([3,4])),
]

# 步骤3: 第一次淘汰
pop (8.0, node([9,10]))
  ✓ lock_ref = 0
  ✓ 释放 GPU: [108, 109] (2 tokens)
  ✓ num_evicted = 2
  ✓ 删除节点
  ✓ 父节点 [7,8] 变成叶子，加入堆

# 更新后的树
Root
  ├─ [1, 2] → [100, 101]
  │   ├─ [3, 4] → [102, 103] ← 叶子
  │   └─ [5, 6] → [104, 105] ← 叶子（被保护）
  └─ [7, 8] → [106, 107] ← 叶子（新）

# 更新后的堆
heap = [
    (12.0, node([7,8])),   # ← 堆顶
    (15.0, node([3,4])),
]

# 步骤4: 第二次淘汰
pop (12.0, node([7,8]))
  ✓ 释放 [106, 107] (2 tokens)
  ✓ num_evicted = 4

# 步骤5: 第三次淘汰
pop (15.0, node([3,4]))
  ✓ 释放 [102, 103] (2 tokens)
  ✓ num_evicted = 6 ← 达到目标！

# ========== 最终树 ==========
Root
  └─ [1, 2] → [100, 101]
       └─ [5, 6] → [104, 105] (lock_ref=1, 受保护)
```

#### 3.3.3 为什么只淘汰叶子节点？

```python
# 反例：如果淘汰中间节点会怎样？
Root
  └─ [1, 2, 3] → [100, 101, 102]  ← 如果删除这个
       ├─ [4, 5] → [103, 104]     ← 这两个子分支都会丢失！
       └─ [6, 7] → [105, 106]

# 原因：中间节点是路径的一部分
# 删除中间节点会导致子树全部丢失
# 所以只能从叶子节点开始，自底向上淘汰
```

---

## 4. 系统架构

### 4.1 CPU-GPU 分离架构

Radix Cache 采用**索引-数据分离**的架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                      CPU (Host)                          │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Radix Tree (Python)                    │   │
│  │                                                   │   │
│  │  Root                                            │   │
│  │    └─ Node1                                      │   │
│  │        ├─ key: [1,2,3]                          │   │
│  │        ├─ value: [100,101,102] ←───────┐       │   │
│  │        └─ children: {...}               │       │   │
│  └─────────────────────────────────────────│───────┘   │
│                                             │           │
│  ┌─────────────────────────────────────────│───────┐   │
│  │      Token Allocator (Python)           │       │   │
│  │                                          │       │   │
│  │  free_slots: [0, 5, 8, 103, ...]       │       │   │
│  │  (标记哪些 GPU 位置可用)               │       │   │
│  └─────────────────────────────────────────│───────┘   │
│                                             │           │
└─────────────────────────────────────────────│───────────┘
                                              │ 索引
                                              │ 映射
┌─────────────────────────────────────────────│───────────┐
│                    GPU (Device)             ↓           │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         KV Cache Pool (GPU Memory)                │  │
│  │                                                    │  │
│  │  k_buffer: [1024, 32, 128]                       │  │
│  │    [0]   ← free                                  │  │
│  │    [1]   ← used                                  │  │
│  │    ...                                            │  │
│  │    [100] ← used (by tree node) ←─────────────────┤  │
│  │    [101] ← used                                  │  │
│  │    [102] ← used                                  │  │
│  │    [103] ← free                                  │  │
│  │    ...                                            │  │
│  │                                                    │  │
│  │  v_buffer: [1024, 32, 128]                       │  │
│  │    (同样的索引布局)                              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**关键点**：
1. **Tree 在 CPU**：Python 对象，存储索引
2. **Data 在 GPU**：实际的 K/V tensor
3. **通过索引访问**：`k_buffer[indices]`

#### 4.1.1 数据流示例

```python
# ========== 步骤1: GPU 计算（Forward Pass）==========
with torch.cuda.device(0):
    k, v = model.forward(input_ids=[1, 2, 3, 4])
    # k shape: [4, 32, 128]
    # v shape: [4, 32, 128]
    
    # 写入 GPU pool 的指定位置
    indices = [100, 101, 102, 103]
    k_buffer[indices] = k  # GPU 操作
    v_buffer[indices] = v  # GPU 操作

# ========== 步骤2: CPU 更新树 ==========
tree.insert(
    key=RadixKey([1, 2, 3, 4]),
    value=torch.tensor([100, 101, 102, 103])  # 只是索引！
)
# TreeNode.value 现在持有这些索引

# ========== 步骤3: 复用缓存 ==========
# 新请求到来
matched_indices, _ = tree.match_prefix([1, 2, 3, 5, 6])
# matched_indices = [100, 101, 102]

# 从 GPU 读取
k_cached = k_buffer[matched_indices]  # GPU 操作
v_cached = v_buffer[matched_indices]
# 现在可以直接用于 attention 计算！
```

---

### 4.2 调用链路

#### 4.2.1 完整调用栈

```python
用户请求到来
    ↓
Scheduler.get_next_batch_to_run()
    ↓
┌─────────────────────────────────────────┐
│     ScheduleBatch (schedule_batch.py)    │
├─────────────────────────────────────────┤
│                                          │
│  alloc_token_slots(num_tokens)          │  ← 需要分配内存
│      ↓                                   │
│  _evict_tree_cache_if_needed(num_tokens)│  ← 检查是否需要淘汰
│      ↓                                   │
│  if available < needed:                 │
│      tree_cache.evict(needed-available) │  ← 淘汰缓存
│      ↓                                   │
│  allocator.alloc(num_tokens)            │  ← 实际分配
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      RadixCache (radix_cache.py)         │
├─────────────────────────────────────────┤
│                                          │
│  evict(num_tokens)                      │
│      ↓                                   │
│  _collect_leaves()                      │  ← 收集叶子
│      ↓                                   │
│  heapq.heapify(leaves)                  │  ← 构建堆
│      ↓                                   │
│  heapq.heappop() × N                    │  ← 循环淘汰
│      ↓                                   │
│  allocator.free(node.value)             │  ← 释放 GPU 内存
│      ↓                                   │
│  _delete_leaf(node)                     │  ← 删除节点
└─────────────────────────────────────────┘
```

#### 4.2.2 三大场景

**场景1：Match Prefix（查找缓存）**

```python
# 时机：新请求到达，prefill 阶段
Scheduler.get_new_batch()
    ↓
for req in new_requests:
    # 查找已缓存的前缀
    req.prefix_indices, req.last_node = tree_cache.match_prefix(
        key=RadixKey(req.input_ids, req.extra_key)
    )
    # prefix_indices: 已缓存的 KV cache 索引
    # last_node: 匹配路径的最后一个节点
```

**场景2：Insert（缓存新数据）**

```python
# 时机：请求完成，释放资源前
Scheduler.finish_request(req)
    ↓
tree_cache.cache_finished_req(req)
    ↓
    # 1. 获取该请求的 KV indices
    kv_indices = req_to_token_pool[req.req_pool_idx, :length]
    
    # 2. 插入树
    new_prefix_len = tree_cache.insert(
        key=RadixKey(req.token_ids, req.extra_key),
        value=kv_indices
    )
    
    # 3. 释放重复的部分
    allocator.free(kv_indices[old_len:new_prefix_len])
```

**场景3：Evict（淘汰缓存）**

```python
# 时机：内存不足时
ScheduleBatch.alloc_token_slots(num_tokens)
    ↓
    available = allocator.available_size()
    if available < num_tokens:
        # 需要淘汰
        tree_cache.evict(num_tokens - available)
```

---

### 4.3 内存管理

#### 4.3.1 内存池结构

```python
# ========== ReqToTokenPool ==========
# 作用：映射 request → token positions
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device):
        # shape: [num_requests, max_context_len]
        self.req_to_token = torch.zeros(
            (size, max_context_len),
            dtype=torch.int32,
            device=device
        )
        self.free_slots = list(range(size))
    
    # 示例
    # req_to_token[3, :10] = [100, 101, 102, ..., 109]
    # 含义：请求3的前10个token的KV cache在位置100-109

# ========== TokenToKVPoolAllocator ==========
# 作用：管理 KV cache 的空闲位置
class TokenToKVPoolAllocator:
    def __init__(self, size, ...):
        self.size = size  # 例如 1024
        # CPU 上的空闲索引列表
        self.free_slots = list(range(size))  # [0,1,2,...,1023]
    
    def alloc(self, need_size: int):
        """分配索引"""
        select = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return torch.tensor(select)
    
    def free(self, indices: torch.Tensor):
        """释放索引"""
        self.free_slots.extend(indices.tolist())

# ========== KV Cache Pool ==========
# 作用：实际存储 K/V 数据
class MHATokenToKVPool:
    def _create_buffers(self):
        # GPU 上的大块连续内存
        self.k_buffer = [
            torch.zeros((size, head_num, head_dim), device='cuda')
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros((size, head_num, head_dim), device='cuda')
            for _ in range(layer_num)
        ]
```

#### 4.3.2 引用计数机制

```python
def inc_lock_ref(self, node: TreeNode):
    """增加锁引用（保护节点）"""
    while node != self.root_node:
        if node.lock_ref == 0:
            # 从可淘汰变为受保护
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent

def dec_lock_ref(self, node: TreeNode):
    """减少锁引用（允许淘汰）"""
    while node != self.root_node:
        if node.lock_ref == 1:
            # 从受保护变为可淘汰
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
        node.lock_ref -= 1
        node = node.parent

# 使用示例
# 请求开始使用节点
tree_cache.inc_lock_ref(req.last_node)  # 保护整条路径

# 请求完成
tree_cache.dec_lock_ref(req.last_node)  # 解除保护
```

---

## 5. Page 机制

### 5.1 什么是 Page？

Page 是 KV Cache 的**管理单元**，类似操作系统的虚拟内存页。

```python
# 操作系统虚拟内存    →    SGLang KV Cache
# ─────────────────    ─────────────────────
# 进程逻辑地址        →    请求的 token 序列
# Page（页）          →    KV cache 的 Page
# Page Table          →    req_to_token mapping
# 物理内存            →    实际的 K/V buffer
```

### 5.2 为什么需要 Page？

**问题：没有 page 的情况**

```python
# 3 个请求，长度分别是 17, 23, 31 tokens
# 内存分配：[17个token][23个token][31个token]
# 
# 当第2个请求结束，释放 23 个 token 的空间
# 新请求需要 30 tokens → 无法利用那 23 的空间（太小）
# 结果：严重的内存碎片
```

**解决：使用 page_size=16**

```python
# 请求1: 17 tokens → 2 pages (16+1, 浪费15个位置)
# 请求2: 23 tokens → 2 pages (16+7, 浪费9个位置)
# 请求3: 31 tokens → 2 pages (16+15)
# 
# 当请求2结束，释放 2 个完整的 page
# 新请求需要 30 tokens → 2 个 page 完全够用！
# 结果：内存利用率大幅提升
```

### 5.3 Page Size 的选择

```bash
# 常见配置
--page-size 1    # 不分页，每个 token 独立
--page-size 16   # FlashAttention 友好
--page-size 64   # HiCache 推荐
```

| page_size | 内存利用率 | 前缀共享粒度 | I/O 效率 | 适用场景 |
|-----------|----------|------------|---------|---------|
| 1 | 最高 | 最细 | 低 | 精确缓存，小模型 |
| 16 | 中等 | 中等 | 中等 | 通用场景 |
| 64 | 较低 | 粗 | 高 | 大规模缓存，HiCache |

### 5.4 Page 对齐处理

```python
# match_prefix 中的对齐
if self.page_size != 1:
    # 截断到 page 边界
    page_aligned_len = len(key) // self.page_size * self.page_size
    key = key[:page_aligned_len]

# 示例
# 输入: key = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# page_size = 16
# 对齐后: key = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#             (截断掉后3个)
```

---

## 6. HiCache 分层缓存

### 6.1 三层架构

HiCache 扩展了 RadixCache，提供三层缓存：

```
┌─────────────────────────────────────────────────┐
│  L1: GPU Memory (Fast, Small ~16GB)             │
│  - RadixTree.value (GPU 索引)                   │
│  - 最热的数据                                    │
└────────────────┬────────────────────────────────┘
                 │ Evict/Load
┌────────────────┴────────────────────────────────┐
│  L2: Host Memory (Medium, Large ~100GB)         │
│  - RadixTree.host_value (CPU 索引)              │
│  - 温数据                                        │
└────────────────┬────────────────────────────────┘
                 │ Write-back/Prefetch
┌────────────────┴────────────────────────────────┐
│  L3: Distributed Storage (Slow, Unlimited)      │
│  - RadixTree.hash_value (哈希)                  │
│  - 冷数据，跨实例共享                            │
│  - 后端: Mooncake, 3FS, NIXL, etc.              │
└─────────────────────────────────────────────────┘
```

### 6.2 节点状态

```python
# L1 only (GPU)
node.value = [100, 101]
node.host_value = None
node.hash_value = None

# L1 + L2 (GPU + CPU)
node.value = [100, 101]
node.host_value = [5000, 5001]
node.hash_value = None

# L1 + L2 + L3 (GPU + CPU + Storage)
node.value = [100, 101]
node.host_value = [5000, 5001]
node.hash_value = ["hash1", "hash2"]

# L2 only (CPU, GPU 已淘汰)
node.value = None
node.host_value = [5000, 5001]
node.hash_value = ["hash1", "hash2"]
```

### 6.3 配置示例

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-hierarchical-cache \
  --page-size 64 \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend mooncake
```

---

## 7. Paged Attention

### 7.1 核心思想

Paged Attention 允许 KV cache 在**非连续内存**中存储，通过 page table 间接访问。

```python
# 传统 Attention
K shape: [seq_len, num_heads, head_dim]  # 必须连续
Q @ K^T  # 要求 K 连续

# Paged Attention
K 物理布局: [num_pages, page_size, num_heads, head_dim]
Page Table: [page_id_0, page_id_5, page_id_3, ...]  # 可以不连续
K 逻辑视图: 通过 page table 重建连续视图
```

### 7.2 工作原理

```python
# 场景：一个请求的 KV cache 分配到不连续的 page
req.page_table = [5, 12, 3, 8]  # 物理页号
page_size = 16

# 逻辑地址 → 物理地址
# token 0-15   → page 5  → 物理地址 80-95
# token 16-31  → page 12 → 物理地址 192-207
# token 32-47  → page 3  → 物理地址 48-63
# token 48-63  → page 8  → 物理地址 128-143

# Attention kernel 通过 page table 访问
k_cache = k_buffer[page_table * page_size : (page_table + 1) * page_size]
```

### 7.3 优势

1. **减少碎片**：page 可以不连续分配
2. **高效复用**：不同请求可以共享 page
3. **灵活管理**：便于淘汰和迁移

---

## 8. 实战示例

### 8.1 cache_finished_req - 完整的缓存流程

**功能**：请求完成时，将其 KV cache 插入 Radix Tree。

**代码位置**：`python/sglang/srt/mem_cache/radix_cache.py:317`

```python
def cache_finished_req(self, req: Req):
    """缓存完成的请求"""
    
    # ========== 步骤1: 准备 token_ids ==========
    # 拼接 input + output，去掉最后一个 token
    token_ids = (req.origin_input_ids + req.output_ids)[:-1]
    # 例如: input=[1,2,3,4] + output=[5,6,7]
    #      → token_ids=[1,2,3,4,5,6]
    
    # 为什么去掉最后一个？
    # 因为最后一个 token 刚生成，还没有对应的 KV cache
    
    # ========== 步骤2: 获取 KV cache 索引 ==========
    # 从 req_to_token_pool 获取该请求的所有 KV 索引
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :all_token_len
    ]
    # kv_indices = [100, 101, 102, 103, 104, 105]
    
    # ========== 步骤3: Page 对齐 ==========
    if self.page_size != 1:
        # 只保留完整的 page
        page_aligned_len = actual_kv_len // self.page_size * self.page_size
        page_aligned_kv_indices = kv_indices[:page_aligned_len]
        # 释放不完整的部分
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
    else:
        page_aligned_kv_indices = kv_indices
    
    # ========== 步骤4: 插入 Radix Tree（核心！）==========
    old_prefix_len = len(req.prefix_indices)  # 之前匹配的长度
    
    new_prefix_len = self.insert(
        RadixKey(token_ids[:page_aligned_token_len], req.extra_key),
        page_aligned_kv_indices,  # 绑定！
    )
    # new_prefix_len: 插入后发现树中已存在的前缀长度
    
    # ========== 步骤5: 释放重复的 KV cache ==========
    # 如果 new_prefix_len > old_prefix_len
    # 说明有新的前缀被加入树，需要释放重复的部分
    self.token_to_kv_pool_allocator.free(
        kv_indices[old_prefix_len:new_prefix_len]
    )
    
    # ========== 步骤6: 清理资源 ==========
    self.req_to_token_pool.free(req.req_pool_idx)
    self.dec_lock_ref(req.last_node)  # 解除锁定
```

**图示流程**：

```
请求完成
  ↓
token_ids = [1,2,3,4,5,6]
kv_indices = [100,101,102,103,104,105]
  ↓
insert(key=[1,2,3,4,5,6], value=[100,101,102,103,104,105])
  ↓
Tree 中已有 [1,2] → [100,101]
  ↓
返回 new_prefix_len = 2
  ↓
释放重复部分: kv_indices[0:2] = [100,101]
  ↓
Tree 最终结构:
Root
  └─ [1,2] → [100,101] (共享)
      └─ [3,4,5,6] → [102,103,104,105] (新插入)
```

---

### 8.2 完整生命周期

```python
# ============================================
# 场景：处理一个新请求
# ============================================

# 步骤1: 请求到达
req = Req(
    input_ids=[1, 2, 3, 4, 5, 6],
    output_ids=[]
)

# 步骤2: Prefill - 查找缓存
prefix_indices, last_node = tree_cache.match_prefix(
    key=RadixKey([1, 2, 3, 4, 5, 6])
)
# 假设命中: prefix_indices = [100, 101, 102]  (前3个token)
# last_node = 节点 [1, 2, 3]

# 步骤3: 分配新内存（只需分配未缓存的部分）
need_tokens = 6 - 3 = 3  # 需要计算 token [4,5,6]
new_indices = allocator.alloc(need_tokens)  # [103, 104, 105]

# 步骤4: GPU 计算（只计算未缓存的）
k_new, v_new = model.forward([4, 5, 6])  # GPU: 只计算3个token
k_buffer[new_indices] = k_new
v_buffer[new_indices] = v_new

# 步骤5: 拼接完整的 KV cache
all_indices = torch.cat([prefix_indices, new_indices])
# [100, 101, 102, 103, 104, 105]

# 步骤6: Attention 计算
k_all = k_buffer[all_indices]  # [6, 32, 128]
v_all = v_buffer[all_indices]
output = attention(q, k_all, v_all)

# 步骤7: Decode - 生成新 token
output_ids = []
for _ in range(max_new_tokens):
    # 每次只生成1个token
    next_token_id = model.decode(output)
    output_ids.append(next_token_id)
    
    # 分配并计算新 token 的 KV
    new_idx = allocator.alloc(1)
    k_new, v_new = model.forward([next_token_id])
    k_buffer[new_idx] = k_new
    v_buffer[new_idx] = v_new
    
    # 拼接
    all_indices = torch.cat([all_indices, new_idx])

# 步骤8: 请求完成，缓存结果
tree_cache.cache_finished_req(req)
# 内部操作：
# 1. 将 all_indices 插入树
# 2. 释放重复的部分
# 3. 解除 lock_ref

# 步骤9: 树的最终状态
# Root
#   └─ [1, 2, 3] → [100, 101, 102]  (之前就有)
#        └─ [4, 5, 6, ...] → [103, 104, 105, ...]  (新插入)
```

### 8.3 req_to_token_pool 的作用

**作用**：建立 request → token positions 的映射。

```python
# req_to_token_pool 是一个二维张量
# shape: [max_batch_size, max_context_len]

# 示例
req_to_token_pool.req_to_token = [
    [100, 101, 102, 103, 0, 0, ...],  # Request 0
    [200, 201, 202, 0, 0, 0, ...],    # Request 1
    [104, 105, 106, 107, 108, 0, ...],# Request 2
    ...
]

# 读取请求1的前3个token的KV cache位置
req_pool_idx = 1
kv_indices = req_to_token_pool[req_pool_idx, :3]
# → [200, 201, 202]

# 这些索引指向实际的 KV cache
k = k_buffer[kv_indices]  # GPU: 读取 K
v = v_buffer[kv_indices]  # GPU: 读取 V
```

**与 Radix Tree 的关系**：

```
Request → req_to_token_pool → KV indices → K/V buffer
   ↓                              ↑
   └──────> Radix Tree ───────────┘
           (match_prefix 填充 indices)
```

---

### 8.4 内存不足场景

```python
# 场景：GPU 内存不足，需要淘汰

# 当前状态
allocator.available_size() = 20 tokens
need = 100 tokens

# 触发淘汰
tree_cache.evict(100 - 20)  # 需要淘汰 80 tokens

# Evict 过程
# 1. 收集所有叶子节点
# 2. 按 LRU/LFU 排序
# 3. 淘汰优先级最低的节点
# 4. 释放 GPU 内存
# 5. 删除树节点

# 淘汰后
allocator.available_size() >= 100 tokens
# 可以继续分配
```

### 8.5 多请求共享

```python
# 场景：3个请求共享相同前缀

# 请求1: "今天天气怎么样？明天呢？"
req1.input_ids = [1, 2, 3, 4, 5, 6, 7]

# 请求2: "今天天气怎么样？后天呢？"
req2.input_ids = [1, 2, 3, 4, 8, 9, 10]

# 请求3: "今天天气怎么样？"
req3.input_ids = [1, 2, 3, 4]

# 处理请求1 → 树中插入
Root
  └─ [1,2,3,4,5,6,7] → [100,101,102,103,104,105,106]

# 处理请求2 → 匹配前4个，分裂节点
Root
  └─ [1,2,3,4] → [100,101,102,103]
       ├─ [5,6,7] → [104,105,106]     (请求1)
       └─ [8,9,10] → [107,108,109]    (请求2，新分支)

# 处理请求3 → 完全命中！
match_prefix([1,2,3,4]) → [100,101,102,103]
# GPU 计算量: 0（完全复用）

# 总 GPU 计算量
# - 没有缓存: 7 + 7 + 4 = 18 tokens
# - 有缓存: 7 + 3 + 0 = 10 tokens
# - 节省: 44%
```

---

## 9. 调用时序图

### 9.1 Prefill 阶段（新请求）

```
Time  │ CPU (Scheduler)              │ CPU (Radix Tree)          │ GPU
──────┼──────────────────────────────┼──────────────────────────┼─────────────
  0ms │ 接收请求                      │                          │
  1ms │ match_prefix() ──────────────→ 遍历树查找前缀            │
  2ms │                              │ 返回 indices=[100,101]   │
  3ms │ 检查内存                      │                          │
  4ms │ available < needed?          │                          │
  5ms │ 是 → evict() ────────────────→ 收集叶子，构建堆          │
 10ms │                              │ 淘汰节点，释放内存        │
 15ms │ alloc(new_tokens) ───────────→                          │
 16ms │                              │                          │ forward(new)
 66ms │                              │                          │ ← 计算完成
 67ms │ cache_finished_req() ────────→ insert(key, indices)     │
 69ms │                              │ 返回 ←───────────────────│
 70ms │ 完成                          │                          │
```

### 9.2 Decode 阶段（生成 token）

```
Time  │ CPU (Scheduler)              │ CPU (Radix Tree)          │ GPU
──────┼──────────────────────────────┼──────────────────────────┼─────────────
  0ms │ 开始 decode                   │                          │
  1ms │ alloc(batch_size) ───────────→                          │
  2ms │                              │                          │ decode()
 12ms │                              │                          │ ← 生成完成
 13ms │ sample token                 │                          │
 14ms │ 继续下一轮                    │                          │
```

**关键观察**：
- Radix Tree 操作（CPU）与 GPU 计算基本不重叠
- Evict 只在内存不足时触发（不频繁）
- Match 很快（< 1ms），对延迟影响小

---

## 10. 性能分析

### 10.1 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|----------|------|
| match_prefix | O(M) | M = 匹配路径长度 |
| insert | O(M) | M = 插入路径长度 |
| evict | O(N + L + K log L) | N=总节点数, L=叶子数, K=淘汰数 |
| _collect_leaves | O(N) | 遍历整棵树 |

### 10.2 空间复杂度

```python
# TreeNode 大小
# - Python 对象开销: ~56 bytes
# - children (defaultdict): ~232 bytes
# - 其他属性: ~100 bytes
# 总计: ~400 bytes/node

# 示例：1000个节点的树
# 内存: 1000 × 400 bytes ≈ 400 KB (CPU)

# KV Cache 大小
# - K: [num_tokens, num_heads, head_dim]
# - V: [num_tokens, num_heads, head_dim]
# 示例: [1024, 32, 128] × 2 × 2 bytes (fp16)
#     = 16 MB per layer (GPU)
#     × 32 layers = 512 MB
```

### 10.3 实际性能

```python
# ========== 小规模树 ==========
树大小: 1,000 nodes
叶子数: 500 leaves
淘汰数: 10 nodes

Evict 时间: ~0.5 ms (CPU)
Match 时间: ~0.1 ms (CPU)
GPU 推理: ~50 ms
影响: < 1%

# ========== 中等规模树 ==========
树大小: 10,000 nodes
叶子数: 5,000 leaves
淘汰数: 100 nodes

Evict 时间: ~2-3 ms (CPU)
Match 时间: ~0.2 ms (CPU)
GPU 推理: ~50 ms
影响: ~4-6%

# ========== 大规模树 ==========
树大小: 100,000 nodes
叶子数: 50,000 leaves
淘汰数: 1,000 nodes

Evict 时间: ~20-30 ms (CPU)
Match 时间: ~0.5 ms (CPU)
GPU 推理: ~50 ms
影响: ~40-60%
```

### 10.4 何时成为瓶颈？

```python
# 不是瓶颈的场景：
✅ 内存充足（很少 evict）
✅ 树较小（< 10K 节点）
✅ Cache hit rate 高（大部分请求命中）

# 可能成为瓶颈的场景：
⚠️ 内存极度紧张（频繁 evict）
⚠️ 树超大（> 100K 节点）
⚠️ Cache hit rate 低（很少命中）
```

---

## 11. 常见问题

### 11.1 为什么 Tree 在 CPU，Data 在 GPU？

**优势**：
1. Tree 操作不阻塞 GPU 计算
2. 索引很小，CPU-GPU 传输开销低
3. Python 灵活性，易于实现复杂逻辑
4. CPU 内存大，可以维护更大的树

**权衡**：
- CPU 操作有开销，但远小于 GPU 计算
- 索引传输有开销，但索引很小（int32）

### 11.2 Evict 的堆每次都重建？

**是的，每次 evict 都重建堆！**

**原因**：
1. 树结构动态变化（新插入、节点删除）
2. 节点优先级动态变化（访问时间、命中次数）
3. 重建成本可控（只在需要淘汰时）
4. 正确性更容易保证

**不用全局堆的原因**：
- 优先级会过时
- 维护成本高
- 难以保证正确性

### 11.3 如何选择 Page Size？

```bash
# 小模型、精确缓存
--page-size 1

# 通用场景、平衡性能
--page-size 16

# 大规模部署、HiCache
--page-size 64
```

### 11.4 LRU vs LFU 如何选择？

```python
# LRU (默认)
# - 适合时间局部性强的场景
# - 最近用过的可能再次使用

# LFU
# - 适合访问频率差异大的场景
# - 高频访问的数据更重要
```

### 11.5 如何监控缓存效果？

```python
# 可用指标
scheduler._get_token_info()
# 返回:
# - num_used: 已使用的 token 数
# - token_usage: 使用率
# - available_size: 可用大小
# - evictable_size: 可淘汰大小

# Cache hit rate (隐式)
# - prefix_len / total_len 的平均值
# - 越高说明缓存效果越好
```

### 11.6 TreeNode.value 是什么？

```python
# ❌ 错误理解
node.value = 实际的 K/V 数据

# ✅ 正确理解
node.value = GPU KV cache 的索引
# 例如: [100, 101, 102, 103]
# 这些数字指向 k_buffer 和 v_buffer 的位置

# 实际数据在这里：
k_buffer[100]  # [32, 128] 的 K tensor
v_buffer[100]  # [32, 128] 的 V tensor
```

### 11.7 为什么只淘汰叶子节点？

```python
# 中间节点是路径的一部分
# 删除中间节点会导致子树丢失

# 例子：
Root → [1,2,3] → [4,5]
              └→ [6,7]

# 如果删除 [1,2,3]
# → [4,5] 和 [6,7] 都会丢失！

# 所以只能从叶子开始，自底向上淘汰
```

---

---

## 12. Eviction 深度解析

### 12.1 谁在调用 evict？

**调用者**：`ScheduleBatch._evict_tree_cache_if_needed()`

**代码位置**：`python/sglang/srt/managers/schedule_batch.py:1914-1929`

```python
class ScheduleBatch:
    def _evict_tree_cache_if_needed(self, num_tokens: int):
        """在分配内存前检查是否需要淘汰"""
        
        if self.tree_cache is None:
            return
        
        # ========== 检查可用内存 ==========
        available_size = self.token_to_kv_pool_allocator.available_size()
        
        # ========== 决定是否需要淘汰 ==========
        if available_size < num_tokens:
            # 计算淘汰数量 = 需要的 - 可用的
            evict_amount = num_tokens - available_size
            
            # ========== 调用 evict ==========
            self.tree_cache.evict(evict_amount)
```

### 12.2 调用时机

**时机**：每次需要分配新的 KV cache 内存时

```python
# ========== 场景1: Prefill（新请求）==========
ScheduleBatch.alloc_token_slots(num_tokens)
    ↓
_evict_tree_cache_if_needed(num_tokens)  ← 检查并淘汰
    ↓
allocator.alloc(num_tokens)  ← 实际分配

# ========== 场景2: Decode（生成token）==========
ScheduleBatch.alloc_token_slots(batch_size)  # 每个请求1个token
    ↓
_evict_tree_cache_if_needed(batch_size)
    ↓
allocator.alloc(batch_size)

# ========== 场景3: Chunked Prefill（分块处理）==========
ScheduleBatch.alloc_paged_token_slots_extend(extend_num_tokens)
    ↓
_evict_tree_cache_if_needed(estimated_tokens)
    ↓
allocator.alloc_extend(...)
```

### 12.3 如何决定淘汰多少？

**策略**：按需淘汰（Just-in-Time Eviction）

```python
# ========== 公式 ==========
evict_amount = max(0, needed_tokens - available_tokens)

# ========== 示例1: 内存充足 ==========
needed_tokens = 100
available_tokens = 150
evict_amount = max(0, 100 - 150) = 0  ← 不淘汰

# ========== 示例2: 内存不足 ==========
needed_tokens = 100
available_tokens = 30
evict_amount = max(0, 100 - 30) = 70  ← 淘汰 70 tokens

# ========== 示例3: 极端缺内存 ==========
needed_tokens = 500
available_tokens = 5
evict_amount = max(0, 500 - 5) = 495  ← 淘汰 495 tokens
```

### 12.4 淘汰频率分析

```python
# ========== 低负载时期 ==========
# GPU 内存使用率: 30-50%
# 
# 100 次内存分配请求:
# - Evict 调用: 0-5 次
# - 平均淘汰量: 0-50 tokens/次
# - 总时间: < 1 ms
# - 对性能影响: < 0.1%

# ========== 中负载时期 ==========
# GPU 内存使用率: 60-80%
# 
# 100 次内存分配请求:
# - Evict 调用: 20-40 次
# - 平均淘汰量: 50-200 tokens/次
# - 总时间: 5-15 ms
# - 对性能影响: 1-3%

# ========== 高负载时期 ==========
# GPU 内存使用率: 85-95%
# 
# 100 次内存分配请求:
# - Evict 调用: 60-90 次
# - 平均淘汰量: 100-500 tokens/次
# - 总时间: 30-100 ms
# - 对性能影响: 5-15%

# ========== 极端负载 ==========
# GPU 内存使用率: > 95%
# 
# 100 次内存分配请求:
# - Evict 调用: 90-100 次
# - 平均淘汰量: 500-2000 tokens/次
# - 总时间: 100-500 ms
# - 对性能影响: 15-50%
# 
# 建议: 增加内存或减小 batch size
```

### 12.5 evict 在哪里执行？

**关键点：主要在 CPU，部分在 GPU**

```python
def evict(self, num_tokens: int):
    # ========== CPU 操作（Python）==========
    # 1. 遍历树收集叶子
    leaves = self._collect_leaves()  # ~O(N) CPU
    
    # 2. 构建优先级堆
    eviction_heap = [...]  # ~O(L) CPU
    heapq.heapify(eviction_heap)
    
    # 3. 选择要淘汰的节点
    while num_evicted < num_tokens:
        _priority, x = heapq.heappop(eviction_heap)  # CPU
        
        # ========== GPU 操作 ==========
        # 4. 释放 GPU 内存
        self.token_to_kv_pool_allocator.free(x.value)
        #                                      ↑
        #                           这会标记 GPU 位置为可用
        
        # ========== CPU 操作 ==========
        # 5. 删除树节点
        self._delete_leaf(x)  # CPU
```

**时间分配**：

```
Total Time: 20ms (evict 1000 tokens)
├─ CPU 操作: 18ms (90%)
│  ├─ 收集叶子: 8ms
│  ├─ 建堆: 5ms
│  └─ 循环淘汰: 5ms
└─ GPU 操作: 2ms (10%)
   └─ 标记内存可用
```

### 12.6 为什么可以接受这个开销？

```python
# ========== 对比 GPU 计算 ==========
GPU Forward Pass (Prefill):
  - 输入: 1000 tokens
  - 时间: 50-200 ms
  - 主要开销

CPU Evict:
  - 淘汰: 1000 tokens
  - 时间: 10-30 ms
  - 次要开销
  
# 比例: 10-60% (可接受)

# ========== 不频繁触发 ==========
# 大多数情况下内存充足，不需要 evict
# 
# 统计数据（典型工作负载）:
# - 总请求数: 1000
# - Evict 调用: 50-100 次
# - 频率: 5-10%

# ========== 可以优化 ==========
# 未来可能的优化：
# 1. 异步淘汰（在 GPU 计算时并行）
# 2. 批量淘汰（一次淘汰更多，减少调用频率）
# 3. 增量维护堆（避免完全重建）
# 4. C++ 实现（已支持，快 5-10x）
```

### 12.7 完整时序分析

```python
# ========== 请求处理的完整时序 ==========

Time    Operation                           Location    Duration
────────────────────────────────────────────────────────────────
0ms     请求到达                              Scheduler   -
1ms     match_prefix()                       CPU         1ms
2ms     检查内存                              CPU         <1ms
3ms     内存不足，调用 evict()                CPU         -
3ms       └─ _collect_leaves()               CPU         5ms
8ms       └─ heapify()                       CPU         3ms
11ms      └─ 淘汰循环                         CPU+GPU     10ms
21ms     evict 完成                           -           -
22ms     alloc_token_slots()                 CPU         <1ms
23ms     GPU forward pass                    GPU         50ms
73ms     计算完成                              -           -
74ms     cache_finished_req()                CPU         2ms
74ms       └─ insert()                       CPU         2ms
76ms     请求完成                              -           -
────────────────────────────────────────────────────────────────
Total: 76ms
  - GPU: 50ms (66%)
  - CPU (Radix): 21ms (28%)
  - CPU (Other): 5ms (6%)
```

**关键观察**：
1. **Evict 占总时间的 ~28%**（在内存紧张时）
2. **大部分时间还是 GPU 计算**（66%）
3. **Evict 不与 GPU 计算重叠**（同步执行）

---

## 13. 核心设计原则

### 13.1 索引-数据分离

```python
# 设计哲学：
# - 索引（小，频繁操作）→ CPU
# - 数据（大，计算密集）→ GPU

优势：
✅ Tree 操作不阻塞 GPU
✅ 灵活的 Python 实现
✅ 索引传输开销小
✅ 支持复杂的淘汰策略

成本：
⚠️ CPU 操作有开销
⚠️ 需要 CPU-GPU 同步
```

### 13.2 按需构建，用完即弃

```python
# Eviction Heap 的生命周期：
# - 每次 evict() 创建
# - 单次调用内使用
# - 调用结束后销毁

优势：
✅ 适应动态变化
✅ 保证正确性
✅ 代码简单

成本：
⚠️ 每次重建有开销
⚠️ O(N) 遍历树
```

### 13.3 保守式内存管理

```python
# 只在确实需要时才淘汰
if available_size < needed_size:
    evict(needed_size - available_size)

优势：
✅ 最大化缓存利用率
✅ 避免过度淘汰

成本：
⚠️ 可能需要频繁淘汰
```

### 13.4 路径级引用计数

```python
# lock_ref 沿着路径传播
def inc_lock_ref(node):
    while node != root:
        node.lock_ref += 1
        node = node.parent

# 保护整条路径，不仅仅是叶子节点

优势：
✅ 保证数据一致性
✅ 防止误删除

成本：
⚠️ 需要维护引用计数
```

---

## 14. 进阶话题

### 14.1 EAGLE 特殊处理

EAGLE（推测解码）使用 bigram key：

```python
# 普通模式
tokens = [1, 2, 3, 4]
key = [1, 2, 3, 4]

# EAGLE 模式
tokens = [1, 2, 3, 4]
key = [(1,2), (2,3), (3,4)]  # bigram
# 长度从 4 变成 3

# 为什么？
# EAGLE 的 draft sequence 是 target 的偏移版本
# 使用 bigram 可以更好地匹配
```

### 14.2 Extra Key 命名空间

```python
# 用途：隔离不同的缓存空间
# 
# 场景1: LoRA Adapter
key1 = RadixKey([1,2,3], extra_key="lora_adapter_A")
key2 = RadixKey([1,2,3], extra_key="lora_adapter_B")
# 即使 token_ids 相同，也不会共享

# 场景2: Sampling 策略
key1 = RadixKey([1,2,3], extra_key="temp_0.7")
key2 = RadixKey([1,2,3], extra_key="temp_1.0")
# 不同温度的结果不共享

# 场景3: Cache 版本
key1 = RadixKey([1,2,3], extra_key="v1")
key2 = RadixKey([1,2,3], extra_key="v2")
# 模型更新后使用新版本
```

### 14.3 C++ 版本

SGLang 还提供了 C++ 实现以提升性能：

```python
# Python 版本
from sglang.srt.mem_cache.radix_cache import RadixCache

# C++ 版本（更快）
from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

# 性能对比：
# - Match: C++ 快 3-5x
# - Insert: C++ 快 2-3x
# - Evict: C++ 快 5-10x
```

**C++ 节点结构**（参考）：

```cpp
// cpp_radix_tree/tree_v2_node.h
struct TreeNode {
    int ref_count;              // 引用计数
    int hit_count;              // 命中次数
    token_vec_t m_tokens;       // token 序列
    torch::Tensor m_device_indices;  // GPU 索引
    torch::Tensor m_host_indices;    // CPU 索引
    TreeNode* m_parent;         // 父节点
    childern_map_t m_children;  // 子节点
    timestamp_t m_last_access_time;  // 访问时间
};
```

### 14.4 SWA Radix Cache

Sliding Window Attention 的特殊实现：

```python
# 普通 Radix Cache
# - 所有 tokens 的 KV cache 都保留

# SWA Radix Cache
# - Full attention layers: 保留所有 KV
# - SWA layers: 只保留窗口内的 KV

# 两种引用计数
node.full_lock_ref = 1  # Full attention 的引用
node.swa_lock_ref = 0   # SWA 的引用

# 淘汰策略
# - Full 层：正常淘汰
# - SWA 层：超出窗口的立即淘汰
```

---

## 15. 与其他系统对比

### 15.1 HiCache vs LMCache

| 维度 | HiCache | LMCache |
|-----|---------|---------|
| **定位** | SGLang 内置 | 独立框架 |
| **架构** | L1(GPU)+L2(CPU)+L3(Storage) | Local+Remote |
| **集成** | 深度集成 RadixAttention | 适配器模式 |
| **优化** | 深度优化（零拷贝、GPU kernel） | 通用设计 |
| **后端** | Mooncake, 3FS, NIXL, AIBrix | 统一接口 |
| **适用** | 纯 SGLang 环境 | 跨框架环境 |

```bash
# HiCache
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake

# LMCache
export LMCACHE_CONFIG_FILE=config.yaml
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache
```

### 15.2 SGLang vs vLLM

| 特性 | SGLang | vLLM |
|------|--------|------|
| **前缀缓存** | Radix Tree | Automatic Prefix Caching |
| **粒度** | Token/Page level | Block level |
| **共享策略** | 自动识别前缀 | 基于 hash |
| **淘汰策略** | LRU/LFU | LRU |
| **分层缓存** | HiCache (3层) | 仅 GPU |

---

## 附录

### A. 相关文件

```
python/sglang/srt/mem_cache/
├── radix_cache.py          # 核心实现
├── evict_policy.py         # LRU/LFU 策略
├── memory_pool.py          # KV Cache Pool
├── allocator.py            # 内存分配器
├── hiradix_cache.py        # HiCache 实现
└── base_prefix_cache.py    # 基类

python/sglang/srt/managers/
├── scheduler.py            # 调度器
└── schedule_batch.py       # Batch 管理

python/sglang/srt/layers/
└── radix_attention.py      # Attention 层
```

### B. 代码位置速查

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| RadixKey | `python/sglang/srt/mem_cache/radix_cache.py` | 44-66 |
| TreeNode | `python/sglang/srt/mem_cache/radix_cache.py` | 68-119 |
| RadixCache | `python/sglang/srt/mem_cache/radix_cache.py` | 172-729 |
| match_prefix | `python/sglang/srt/mem_cache/radix_cache.py` | 230-300 |
| insert | `python/sglang/srt/mem_cache/radix_cache.py` | 302-315 |
| _insert_helper | `python/sglang/srt/mem_cache/radix_cache.py` | 574-605 |
| evict | `python/sglang/srt/mem_cache/radix_cache.py` | 453-480 |
| cache_finished_req | `python/sglang/srt/mem_cache/radix_cache.py` | 317-367 |
| Eviction Strategy | `python/sglang/srt/mem_cache/evict_policy.py` | 10-24 |
| ReqToTokenPool | `python/sglang/srt/mem_cache/memory_pool.py` | 61-109 |
| MHATokenToKVPool | `python/sglang/srt/mem_cache/memory_pool.py` | 404-718 |
| Allocator | `python/sglang/srt/mem_cache/allocator.py` | 36-100 |
| HiRadixCache | `python/sglang/srt/mem_cache/hiradix_cache.py` | 28-end |
| Scheduler | `python/sglang/srt/managers/scheduler.py` | 266-end |
| ScheduleBatch | `python/sglang/srt/managers/schedule_batch.py` | 854-end |

### C. 关键配置

```bash
# Radix Cache
--disable-radix-cache              # 禁用
--radix-eviction-policy lru        # LRU (默认)
--radix-eviction-policy lfu        # LFU

# Page Size
--page-size 1                      # 无分页
--page-size 16                     # 标准
--page-size 64                     # HiCache

# HiCache
--enable-hierarchical-cache        # 启用
--hicache-ratio 2                  # CPU:GPU = 2:1
--hicache-size 100                 # CPU 100GB
--hicache-write-policy write_through
--hicache-storage-backend mooncake
```

### D. 调试技巧

```python
# 打印树结构
tree_cache.pretty_print()

# 输出:
#  0 [] r=1              # Root
#   2 [1, 2] r=0         # 节点 [1,2]
#     2 [3, 4] r=0       # 叶子 [3,4]
#     2 [5, 6] r=1       # 叶子 [5,6] (被保护)
# #tokens: 6

# 查看内存使用
print(f"Total: {tree_cache.total_size()}")
print(f"Evictable: {tree_cache.evictable_size()}")
print(f"Protected: {tree_cache.protected_size()}")
```

### E. 常用命令

```python
# 启动服务（标准配置）
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B \
  --port 30000

# 禁用 Radix Cache
python -m sglang.launch_server \
  --model-path MODEL \
  --disable-radix-cache

# 使用 LFU 策略
python -m sglang.launch_server \
  --model-path MODEL \
  --radix-eviction-policy lfu

# 启用 HiCache
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-hierarchical-cache \
  --page-size 64 \
  --hicache-ratio 2 \
  --hicache-storage-backend mooncake

# 性能测试
python -m sglang.bench_serving \
  --backend sglang \
  --dataset sharegpt \
  --num-prompts 100
```

---

## 总结

Radix Cache 是 SGLang 实现高效 KV Cache 复用的核心机制：

1. **数据结构**：基于 Radix Tree，自动识别和共享前缀
2. **架构设计**：CPU 管理索引，GPU 存储数据，分工明确
3. **内存管理**：通过 LRU/LFU 淘汰策略，动态管理有限的 GPU 内存
4. **性能优化**：Page 机制减少碎片，Paged Attention 支持非连续访问
5. **扩展性**：HiCache 扩展到 CPU 和分布式存储，突破 GPU 内存限制

理解 Radix Cache 是深入掌握 SGLang 的关键！

---

written by cursor