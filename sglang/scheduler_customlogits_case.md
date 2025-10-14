```
┌─────────────────────────────────────────────────────────────┐
│ 1. 用户入口层（主进程）                                      │
└─────────────────────────────────────────────────────────────┘
HTTP Server / Engine API / Runtime
    ↓ text/input_ids + sampling_params + image_data ...
    
┌─────────────────────────────────────────────────────────────┐
│ 2. TokenizerManager（主进程）                                │
└─────────────────────────────────────────────────────────────┘
GenerateReqInput（原始请求对象）
    ↓ 
参数验证（检查开关：custom_logit_processor/return_hidden_states等）
    ↓
分词（Tokenization）+ 多模态预处理
    ├─ text → input_ids
    ├─ 图像/视频 → 视觉特征 (mm_inputs)
    └─ 合并为 TokenizedGenerateReqInput
    ↓
通过 ZeroMQ IPC 发送元信息给 Scheduler 子进程
    ↓ (只传请求元数据，不跨进程拷贝大张量)
    
┌─────────────────────────────────────────────────────────────┐
│ 3. Scheduler（调度器子进程）                                 │
└─────────────────────────────────────────────────────────────┘
创建 Req 对象（rid, input_ids, sampling_params, ...）
    ↓
加入等待队列 (waiting_queue)
    ↓
调度决策
    ├─ Prefix Cache 匹配（RadixAttention 树查找共享前缀）
    ├─ 优先级排序 & 负载均衡
    └─ KV Cache 空间检查（req_to_token_pool 分配槽位）
    ↓
批合并（Continuous Batching）
    ├─ 多个 Req → 一个 ForwardBatch
    ├─ 确定 ForwardMode：EXTEND（prefill）或 DECODE（decode）
    └─ 构建张量：input_ids, positions, seq_lens, out_cache_loc
    ↓
构建 SamplingBatchInfo（采样参数批量合并）
    ├─ 合并 temperature, top_k, top_p, penalties
    └─ 反序列化 custom_logit_processor（如果有）
    ↓
通过 ZeroMQ 发送 ForwardBatch 元信息给 TP Worker
    ↓ (大张量在 worker 内部直接构造，不跨进程传输)
    
┌─────────────────────────────────────────────────────────────┐
│ 4. TP Worker（模型推理子进程，可多卡 Tensor Parallel）       │
└─────────────────────────────────────────────────────────────┘
ModelRunner.forward_batch(forward_batch)
    ↓
Embedding 层
    ↓ input_ids → hidden_states [batch_size, seq_len, hidden_dim]
    
逐层 Transformer 前向（循环 num_layers）
    ┌─────────────────────────────────────────┐
    │  单层 Transformer                        │
    ├─────────────────────────────────────────┤
    │  Attention                               │
    │    ├─ 计算 Q, K, V                       │
    │    ├─ ★ 写入 KV Cache ★                  │
    │    │   token_to_kv_pool.set_kv_buffer(  │
    │    │       layer, cache_loc, k, v       │
    │    │   )                                 │
    │    │                                   │
    │    ├─ 从 token_to_kv_pool 读取历史 KV   │
    │    └─ Attention Kernel（FlashAttn等）   │
    │  MLP / FFN                               │
    │  LayerNorm / Residual                    │
    └─────────────────────────────────────────┘
    ↓
LM Head（最后一层线性投影）
    ↓ hidden_states → logits [batch_size, vocab_size]
    
┌─────────────────────────────────────────────────────────────┐
│ 5. Sampler（采样器，在 TP Worker 内部）                      │
└─────────────────────────────────────────────────────────────┘
Sampler.forward(logits_output, sampling_info)
    ↓
_preprocess_logits(logits, sampling_info)
    ├─ 应用 custom_logit_processor（如果有）
    │   遍历 processor，就地修改 logits[mask]
    └─ NaN 检测 & 修复
    ↓
采样策略链
    ├─ 温度缩放：logits /= temperature
    ├─ Softmax → probs
    ├─ Top-K 过滤
    ├─ Top-P（nucleus sampling）
    ├─ Min-P 过滤
    └─ Repetition/Frequency/Presence penalties
    ↓
生成 next_token_ids
    ├─ Greedy：torch.argmax(logits)
    └─ 随机：torch.multinomial(probs)
    ↓
如需 logprobs → 计算并保存
    ↓
通过 ZeroMQ 返回 Scheduler
    ↓ (token_ids + logprobs + meta_info)
    
┌─────────────────────────────────────────────────────────────┐
│ 6. Scheduler 更新状态                                        │
└─────────────────────────────────────────────────────────────┘
接收 TP Worker 返回的 token_ids
    ↓
更新 Req 对象
    ├─ output_ids.append(next_token_id)
    ├─ 更新 KV cache 占用（seq_len++）
    └─ 检查停止条件
        ├─ 遇到 EOS token？
        ├─ 达到 max_tokens？
        └─ 匹配 stop_str（字符串前缀匹配）？
    ↓
未结束 → 重新加入下轮批次（ForwardMode.DECODE）
    ├─ 只需推理 1 个新 token（autoregressive）
    └─ 复用已缓存的 KV
已结束 → 标记 finished，准备返回
    ↓
发送 token_ids 给 DetokenizerManager
    
┌─────────────────────────────────────────────────────────────┐
│ 7. DetokenizerManager（解码器子进程）                        │
└─────────────────────────────────────────────────────────────┘
接收新增 token_ids（增量）
    ↓
Tokenizer.decode()
    ├─ 跳过 special tokens（如配置）
    ├─ 增量解码（只解码新 token，避免重复）
    └─ 字符串级 stop_str 匹配（更精确）
    ↓
生成 decoded_text（新增文本片段）
    ↓
通过 ZeroMQ 返回 TokenizerManager
    
┌─────────────────────────────────────────────────────────────┐
│ 8. TokenizerManager 返回用户（主进程）                       │
└─────────────────────────────────────────────────────────────┘
TokenizerManager 接收解码结果
    ↓
构造响应对象
    ├─ text: decoded_text
    ├─ meta_info: {prompt_tokens, completion_tokens, finish_reason, ...}
    └─ logprobs（如果请求）
    ↓
返回给用户
    ├─ stream=True  → SSE 流式推送
    │   "data: {text: '...', meta_info: {...}}\n\n"
    │   ...
    │   "data: [DONE]\n\n"
    └─ stream=False → 一次性返回完整 JSON
        {text: '...', meta_info: {...}}
```
