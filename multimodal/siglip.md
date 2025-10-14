### SigLIP 高层概念（High-level Idea）

SigLIP（Sigmoid Loss for Language-Image Pretraining）是谷歌提出的对比学习视觉-语言预训练框架。它在预训练阶段使用“图-文对”和基于 sigmoid 的成对损失学习一个强大的视觉编码器；在多模态 LLM（如 VILA、Ovis）中，通常只复用其中的视觉编码器，将图像转换为可供语言模型理解的序列特征。

参考：`google/siglip-*`（HF 模型族）

---

#### 预训练阶段（与我们推理使用解耦）
```
┌─────────────────────────────────────────┐
│  SigLIP 预训练（Google 的工作）         │
├─────────────────────────────────────────┤
│                                         │
│  输入：图文对                           │
│  ┌─────────┐        ┌─────────┐        │
│  │  🖼️图像  │        │  📝文本  │        │
│  └────┬────┘        └────┬────┘        │
│       ↓                  ↓             │
│  ┌─────────┐        ┌─────────┐        │
│  │ 图像编码器│        │ 文本编码器│       │
│  │  (ViT)  │        │(BERT-like)│      │
│  └────┬────┘        └────┬────┘        │
│       ↓                  ↓             │
│   [1152维]            [1152维]         │
│       └──────┬───────┘                 │
│              ↓                         │
│      计算相似度 & Sigmoid Loss          │
│      (这是训练目标)                    │
│                                         │
│  训练完成后，发布两个编码器到 HF       │
└─────────────────────────────────────────┘
```
- 输入：图像与文本成对数据（大规模）。
- 结构：图像编码器（ViT）+ 文本编码器（Transformer）。
- 目标：使用 Sigmoid 成对损失独立优化每个图文对（相较 CLIP 的 softmax 对比损失，减少 batch 依赖、训练更稳定高效）。
- 产出：两个编码器的权重，其中视觉编码器常被下游多模态任务直接复用。

> 该阶段发生在预训练时，与我们在推理/服务阶段的代码无直接耦合。

---
```
#### 推理/集成阶段（多模态 LLM 中的角色）
┌─────────────────────────────────────────┐
│  多模态模型推理（我们的任务）           │
├─────────────────────────────────────────┤
│                                         │
│  输入：图像 + 问题                      │
│  ┌─────────┐                            │
│  │  🖼️图像  │   "图里有什么？"           │
│  └────┬────┘                            │
│       ↓                                 │
│  ┌─────────────────┐                    │
│  │ SigLIP 图像编码器 │  ← 只用这个！     │
│  │  (冻结/微调)     │                   │
│  └────┬────────────┘                    │
│       ↓                                 │
│   [576, 1152]  视觉特征                 │
│       ↓                                 │
│  ┌─────────┐                            │
│  │投影层MLP │  对齐维度                  │
│  └────┬────┘                            │
│       ↓                                 │
│   [81, 1536]                            │
│       ↓                                 │
│  ┌───────────────┐                      │
│  │ Qwen2/Qwen3   │  语言模型            │
│  │  (LLM)        │                     │
│  └───────┬───────┘                      │
│          ↓                              │
│     "这是一只猫..."                     │
│                                         │
│  ❌ 没有文本编码器                      │
│  ❌ 没有 Sigmoid Loss                   │
│  ❌ 不做图文匹配                        │
```
- 只使用“视觉编码器”将图像变为 patch 序列特征；不涉及文本编码器与对比损失。
- 典型数据流：
  1) 将固定分辨率图像（如 384×384）用卷积做“分块+投影”。
  2) 为每个 patch 添加可学习的位置编码（编码的是“块的位置”，不是块内像素位置）。
  3) 通过多层 Transformer Encoder（双向注意力，Pre-LN）提炼全局语义。
  4) 输出形如 `[num_patches, hidden_dim]` 的视觉 token 序列，供投影层对齐后送入 LLM。

少量示例代码（概念示意）：

```python
# 1) 卷积分块 + 线性投影（等价于切块后线性层）
patch_embeds = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)(image)
# 2) 展平为序列并加位置编码
seq = patch_embeds.flatten(2).transpose(1, 2) + pos_embed(position_ids)
# 3) Transformer Encoder（双向注意力，非因果）
features = encoder(seq)  # [batch, num_patches, hidden_dim]
```

关键要点：
- **每个 patch 对应固定的 16×16 像素区域（示例配置）**，卷积核学习块内的空间模式；位置编码标识“块在整张图中的位置”。
- **无 CLS token**（常见的 SigLIP 视觉分支直接返回所有 patch 特征）。
- **Encoder-only（双向）注意力**，适于理解整幅图的全局与局部关系；文本生成由下游 LLM 完成。

---

#### 与 CLIP 的宏观差异
- 训练目标：CLIP 用 softmax 对比损失（强 batch 依赖）；SigLIP 用 sigmoid 成对损失（弱 batch 依赖、训练更稳）。
- 架构习惯：CLIP 常用 CLS 聚合；SigLIP 更倾向直接使用所有 patch 序列或做池化。
- 下游使用：两者在下游多模态中都多以“视觉编码器”身份被复用；SigLIP 在小中型配置上常有更优的训练效率与性能折中。

---

#### 在 SGLang/VILA 集成中的位置
- 文件：`python/sglang/srt/models/siglip.py`
- 对外暴露：`SiglipVisionModel` 返回 `[batch, num_patches, hidden_dim]` 的视觉序列特征。
- 上游：多模态处理器完成图像预处理（尺寸、归一化）。
- 下游：投影层将视觉维度对齐到 LLM 隐藏维度，再与文本 token 融合输入 LLM。

---

#### 一句话总结
SigLIP 在预训练中用“图文成对 + Sigmoid 损失”学到强表征；在下游多模态中，我们把它当作高质量的 ViT 视觉编码器：把固定分辨率图像分成 patch，得到带位置的视觉 token 序列，经 Transformer 编码后交由 LLM 理解与生成。



---

### NaViT

NaViT（Native Variable Input Transformer）的目标是在“保持固定 patch 尺寸”的前提下，直接处理任意分辨率/纵横比的图像，避免统一 resize 带来的信息损失与失真。[NaViT 论文](https://arxiv.org/abs/2307.06304)；Ovis2.5 使用了 NaViT 思路实现原生分辨率感知与思考模式控制，[Ovis2.5-2B 模型](https://huggingface.co/AIDC-AI/Ovis2.5-2B)。

- 固定 patch、可变序列
  - 保持 `patch_size` 不变（如 16×16），用 `Conv2d(kernel=stride=patch_size)` 做“分块+投影”。
  - 每个 patch 的向量维度固定（如 1152），变化的是 patch 数量 `L=(H/ps)×(W/ps)`。

- 位置编码的尺度自适应
  - 使用可插值的二维绝对位置编码，或相对/旋转位置编码（2D RoPE 等），确保不同网格大小下位置语义一致。
  - 边界对齐常见做法：pad 到 patch 对齐，并以 mask 屏蔽无效块。

- Patch-n-Pack 打包与注意力遮罩
  - 将不同分辨率图片的 patch 序列打包到同一 batch，构造“块对角（block-diagonal）”注意力 mask，防止跨图注意力；同一图内部保持全局双向注意力。
  - 通过“packing 预算”限制总 patch 数（如 `max_pixels`/`max_seq_len`/网格锚点），平衡细节与吞吐、显存。

- 推理侧调优开关（常见）
  - `max_pixels`/`max_seq_len`：限制 patch 总量，控制显存与延迟。
  - `enable_packing`：启用跨样本打包提升吞吐；配合 block-diag mask。
  - `grid_pinpoints`：离散候选网格，兼顾比例保持与预算可控。

极少示例代码（概念示意）：

```python
# 1) 固定维度投影：每个 patch → 固定 embed_dim
seq = nn.Conv2d(3, C, kernel_size=ps, stride=ps)(img)  # [B, C, H/ps, W/ps]
seq = seq.flatten(2).transpose(1, 2)                   # [B, L, C], L 可变
# 2) 位置编码（2D 可插值/相对编码）+ 3) 打包遮罩（块对角 mask）
features = encoder(seq, attn_mask=block_diag_mask)
```

要点回顾：
- 不 resize：靠固定 patch 卷积与可变序列长度直接“吃”原生分辨率；位置编码做尺度自适应。
- 不混图：靠块对角注意力 mask 保证跨样本不互相“看见”。
- 不失控：靠 `max_pixels`/packing 预算限制计算与显存。
