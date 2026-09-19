---
datasets:
- DreamStarPaint/MOSS
- Moemuu/Muice-Dataset
frameworks: PyTorch
license: CC_BY_4.0
domain: nlp
tags:
- 文本生成模型
- 物理模型
tasks:
- text-generation
language:
- zh
- en
studios:
- xuanxixue/AetherMind-V5-300M-Instruct
---
# PMNN 文本生成模型（物理矩阵神经网络 · 通用底层架构）

<p align="center">
<a href="https://modelscope.cn/studios/xuanxixue/AetherMind-V5-300M-Instruct" target="_blank">🚀 Online Demo</a> |
<a href="https://modelscope.cn/models/xuanxixue/AetherMind-V5-300M-Instruct" target="_blank">📦 Model</a>
</p>

按 `docx/PMNN物理矩阵神经网络通用底层架构.md`（v0.1 正式稿）落地的可训练文本模型。
与 Transformer 对位但**非统计学**：用 Kuramoto 相位同步替代注意力、共振-驻波 FNN 替代前馈、
漂移场 + 热力学 + 时间场作为显式物理约束，输出端融合 ELF 连续流匹配。

注意当前为实验验证阶段刚过

---

> **📦 这个目录是干什么的（先看这里）**
>
> 本目录（解压后为 `pmnn_text_kd/`）是 **PMNN ← MiniCPM5-2B 知识蒸馏**的**独立上云包**，
> 与 300M 预训练那轮的 `pmnn_text/` 是两条线，别混。
>
> - **先读包内 `CLOUD_GUIDE.md`**（第 1~9 步照顺序做；含症状对照表与"绝对不要这么干"）
> - 一条命令入口：`bash scripts/run_cloud_kd.sh`，`STAGE=where|probe|prepare|losses|smoke|l1|l2|all`
> - **生效凭证**：日志开头必须有 `[cloud-kd] 补丁版本: kd-guard vN (日期)`，没有就是跑了旧脚本
> - 为什么"架构完全不同也能蒸馏"→ `docx/PMNN_蒸馏方案_MiniCPM5教师.md` 第一节
> （**PMNN 内部 KuramotoSync 算出的 w 与 Transformer 注意力同构，所以注意力对齐不需要投影层**）
>
> 下面第 1~9 节是 **300M 预训练**那轮的工程说明，保留作为模型本体的背景资料。

---

## 1. 目录与路径规范

```
D:\AetherMind-V5\ ← 工作区根目录
├── AetherMind-V5\
│ └── pmnn_text\ ← 本工程（代码只放这里，不与 docx/datasets 混放）
│ ├── paths.py 路径解析层（CLI > 环境变量 > 自动探测）
│ ├── config.py 超参 + 规模预设
│ ├── data_utils.py 词表 + memmap 数据集
│ ├── prepare_data.py 预处理：jsonl → tokens.bin
│ ├── train.py 训练
│ ├── generate.py 生成
│ ├── plot_train.py 曲线可视化
│ ├── smoke_test.py 冒烟测试（不读数据）
│ └── scripts\ 启动脚本（.bat 纯 ASCII / .sh）
├── datasets\ 数据区
└── docx\ 文档区
```

**路径参数化**：所有脚本的路径都走三级解析，云端零改代码。

| 环境变量 | 含义 | 默认 |
|---|---|---|
| `PMNN_ROOT` | 工程根 | 脚本所在目录 |
| `PMNN_DATA_DIR` | 原始 jsonl 目录 | 自动向上查找 `datasets/03_dialogue_clean` |
| `PMNN_OUT_DIR` | 输出根 | 工程根 |
| `PMNN_TOKENIZER` | 词表 | `<out>/data/tokenizer.json` |
| `PMNN_PREPARED_DIR` | 预处理产物 | `<out>/prepared` |
| `PMNN_CKPT_DIR` | checkpoint | `<out>/checkpoints` |
| `PMNN_LOG_DIR` | 日志 | `<out>/logs` |

命令行同名参数（`--data_dir` 等）优先级最高。

---

## 2. 快速开始（本地 RTX 3050 4GB）

```bash
cd AetherMind-V5/pmnn_text
PY=C:/Python312/python.exe

# ① 预处理：流式 tokenize 落盘（内存恒定，几分钟）
$PY prepare_data.py --max_seq_len 128

# ② 冒烟：不读数据，验证前向/反向/生成/流匹配
$PY smoke_test.py --preset tiny

# ③ 训练（预设与步数）
$PY train.py --preset small --steps 3000 --log_every 20

# ④ 生成
$PY generate.py --prompt "你好呀，今天天气怎么样"

# ⑤ 画曲线
$PY plot_train.py
```

Windows 一键：`scripts\run_local.bat 100m 3000`
Linux 云端一键：`bash scripts/run_cloud.sh`

---

## 3. 规模预设与实测显存

4GB 显存下实测（batch=4, seq=128，含优化器状态）：

| 预设 | 参数量 | 优化器 | 峰值显存 | 建议 |
|---|---|---|---|---|
| `tiny` | 2.7M | AdamW | ~0.05 GB | 冒烟测试 |
| `small` | 16.2M | AdamW | 0.33 GB | 本地默认，可加到 batch 32+ |
| `100m` | 88.2M | AdamW | 1.67 GB | **本地架构验证主力** |
| `300m` | 187.6M | AdamW | 3.49 GB (87%) | 太紧 |
| `300m` | 187.6M | Adafactor | 2.24 GB (56%) | ✅ 推荐 |
| `1b` | ~1.1B | Adafactor | — | 仅云端（约 100 GPU 小时） |

先测显存在不 OOM 再正式跑：

```bash
$PY train.py --preset 100m --dry_run --batch 4
```

`--dry_run` 会完整跑「前向 + 反向 + 优化器一步」——因为 AdamW 的动量状态是
**惰性分配**的，只跑前向反向会严重低估显存。

---

## 4. 云端跑 1B（不用改一行代码）

```bash
export PMNN_DATA_DIR=/data/03_dialogue_clean
export PMNN_OUT_DIR=/workspace/pmnn_out

# 大数据集建议更大词表 + 更长序列
python prepare_data.py --max_seq_len 512 --vocab_size 32768
python train.py --preset 1b --steps 300000 \
--batch 16 --grad_accum 8 --num_workers 8 --optimizer adafactor
```

或直接：

```bash
PMNN_DATA_DIR=/data/dialogue PMNN_OUT_DIR=/workspace/out bash scripts/run_cloud.sh
```

---

## 5. 关键工程决策

### 5.1 数据管线改为磁盘 memmap（修复内存崩溃）
旧实现把每条样本 tokenize 成 `list[int]` 常驻内存，20 万条 × 128 token ≈ 数 GB
Python 对象；`num_workers=2` 在 Windows spawn 下再复制一份 → 宿主内存耗尽崩溃。

新实现两阶段：
- **预处理**（`prepare_data.py`）：流式 `encode_batch`，落盘
`tokens.bin`（uint16/uint32 连续 token 流）+ `index.npy`（offset/length）+ `meta.json`。
全程内存恒定，只保留当前 batch。
- **训练**：`np.memmap` 只读映射，进程内存占用≈0，靠页缓存，多 worker 共享同一映射。
`num_workers` 默认 0（Windows 最稳）。

实测 3000 条样本预处理后：`tokens.bin` 241 KB + `index.npy` 48 KB，
读回 token 与原始 tokenize **逐位一致**。

### 5.2 漂移场 einsum 重写
原实现用 4D `[B,N,M,D]` 张量算距离，O(N·M·D) 内存。
改为 `dist² = ‖x‖² + ‖y‖² − 2xyᵀ`（einsum），降到 O(N·M + N·D)：
反向 10.5 s → **0.46 s**（23×），峰值显存 0.39 GB → **0.11 GB**。

### 5.3 mask 泄漏修复（正确性）
左 padding 的 pad token 原本会污染 Kuramoto 平均场、漂移场互斥项、
热力学能量/熵的统计量。现在全部改为 mask 加权：
- `KuramotoSync`：平均场与低秩聚合按有效 token 数归一，pad 相位不更新；
- `DriftField._softmax_pull`：屏蔽 pad 作为吸引中心；
- `Thermodynamics.energy/entropy`：mask 加权平均；
- `order_parameter`：新增 `mask` 参数。

### 5.4 跨位置信息通路：两次尝试（架构级修复）

语言建模要求位置 t 能"按内容取用"它前面的 token。原架构做不到，这是本工程最核心的
一次架构修复，前后共两轮。

#### 5.4.1 第一轮（未奏效）：cumsum 距离衰减耦合

**问题**：原设计的跨位置交互只有「全局长程平均场」与「全局低秩聚合」，两者都是
**置换不变**的——位置 t 拿不到"它前面具体是什么词"的信息，模型只能学会抄自己输入。

（实测：`labels = input_ids` 未右移时 CE 从 9.4 压到 **0.01**，看似收敛极好，实为纯作弊；
labels 修正为右移一位后，CE 卡在 **5.16** 下不去。）

**做法**：给低秩聚合加上指数衰减的因果记忆

```
A_i = Σ_{j≤i} e^{−λ(i−j)} · v_j = e^{−λi} ⊙ cumsum(e^{λj} ⊙ v_j)
```

复杂度 O(N·r)，天然因果，λ = ln2 / `causal_halflife`。

**结果：A/B 对照（同 seed、同数据、400 步）CE 5.1596（ON）vs 5.1622（OFF）——
差异 0.05%，等于没区别。这一轮失败了。**

#### 5.4.2 根因定位：`k ⊙ agg` 根本不是 query-key 交互

回看原实现：

```python
k = self.cpl_key(phi_sin) # [B,N,r]
agg = (v.sum(dim=1, keepdim=True)) # [B,1,r] 或因果版 [B,N,r]
lr_coupling = self.cpl_out(k * agg) # ← Hadamard 积，不是点积！
```

`agg` 只是历史 `v` 的**无差别加权平均**，权重固定为 `e^{−λ(i−j)}`，**与 `k`（自己的键）
完全无关**。所以 `k * agg` 只起到一个逐元素缩放门控的作用，模型**没有任何内容寻址能力**，
无论因果 ON 还是 OFF，能拿到的都只是上下文的 rank-1 摘要。

这解释了 A/B 为何无差异，也解释了 CE 为何卡在 3.59 上不动。

#### 5.4.3 第二轮（当前方案）：内容寻址因果耦合

让耦合强度由**振子状态本身**决定——这正是 Kuramoto 的本意（锁相才强耦合），
也是 `docx/理论.txt` 里写的原始形式 `θ̇ᵢ = ωᵢ + (K/N)·ΣWᵢⱼsin(θⱼ−θᵢ)`
（注意：是**逐对求和**，不是平均场近似）：

```
W_ij = softmax_j( q_i·k_j / √r − λ·(i−j) ), i ≥ j
输出 = W @ sin(θ)
```

- `q, k` 由复值场 `z = a·e^{iθ}` 低秩投影而来 → **耦合强度由内容决定**；
- `−λ(i−j)` 项同时给出光锥因果性与长程衰减；
- softmax 在 fp32 中计算（bf16 下 [B,N,N] 点积易失精），屏蔽项用 `-1e4` 而非 `-inf`
（整行被屏蔽时 softmax 不会出 NaN）。

配置：`--coupling {mean_field, content}`（默认 `content`）、
`--coupling_halflife`（默认 64）、`--coupling_temp`（默认 1.0）。

### 5.5 因果语言建模的 labels 必须右移一位

```python
# 反模式：labels 直接复制输入 → 模型只要抄自己就能把 CE 压到 0
labels = input_ids.clone()

# 正解
labels = input_ids.clone()
labels[:, :-1] = input_ids[:, 1:] # 位置 t 的目标 = 位置 t+1 的 token
labels[:, -1] = -100
labels[input_ids == pad_id] = -100
```

这是最隐蔽的一类 bug：**loss 曲线非常漂亮，但模型什么都没学到**。

### 5.6 物理不变量修复
- 振幅经 LayerNorm 后会出现**负值**（物理上振幅必须 ≥ 0）→ 改为
`softplus(norm(a + ffn))`；
- 相位原本也被 LayerNorm 归一，**破坏 [-π, π] 周期结构** → 取消，相位只由
Kuramoto 内部 wrap 维持。

`smoke_test.py` 对这两条加了断言（振幅非负、序参量 ∈ [0,1]）。

### 5.7 朗之万噪声接入（设计稿 §3.4 的 η(t) 项）

`Thermodynamics.langevin_noise()` 原本是**死代码**——定义了但从未被调用。现在注入点放在
`KuramotoSync.forward` 的积分步：

```
θ_next = θ + dt·f(θ) + √(2T·dt)·N(0,1)
```

⚠️ 关键是标准差必须是 `√(2T·dt)` 而不是 `√(2T)`：那是连续时间 Langevin 方程
`dθ = f·dt + √(2T)·dW`（`dW ~ N(0, dt)`）离散化的结果，漏掉 `dt` 会让噪声量级
随步长错误缩放。温度 `T(t)` 随物理时间退火到 `T_min`，所以后期噪声自然趋于 0（冷却）。

**只在训练期注入**（`self.training`），推理期由采样温度负责随机性。
开关 `--langevin`（默认关）/ `--langevin_gain`。

四道验证（实测）：关→两次前向差异 0；开+train→差异 1.6e+0；开+eval→差异 0；
gain=4→差异 2.2e+0（幅度可按系数放大）。

### 5.8 128 维轨迹瓶颈接入（设计稿 §6 的 ELF 核心技巧）

`self.bottleneck` 原本也是**死代码**：创建了、写进了 `diag`，但没有任何损失使用它 →
参数梯度恒为 `None`。修复后速度场改由瓶颈预测：

```
x_t → 共享物理核心(1 round) → 场[2d] → 128 维瓶颈 + LayerNorm → vel_head → [d]
```

验证：`bottleneck.weight` 梯度 0.212、`bottleneck_norm.weight` 0.026、
`vel_head.weight` 0.844（修复前均为 None）。

**顺带清掉一类"假死通路"**：`content` 模式下 `cpl_key/cpl_val/cpl_out` 这条
mean_field 专用通路不参与前向，会产生 `3×n_blocks` 个无梯度参数（实测 11 个），
在梯度审计里伪装成"死通路"，把真正的问题淹没。现在这两组参数按 `coupling`
**条件创建**——两种模式下活参数梯度均为 0 个死项。

### 5.9 序参量课程：从"单点常数"恢复为"逐层曲线"

设计稿 §7 要求 `L_sync = ‖r(t) − r_target(t)‖²`，其中 `r_target(t) = σ((t−t_c)/τ_c)`。

原实现只取**末层**的 r，且 `t_final` 恒等于 `(n_blocks−1)/(n_blocks−1) = 1.0` →
目标恒为常数 `σ((1−0.5)/0.15) = 0.9656`。**整条"早期层混沌探索 → 晚期层同步整合"
的 K 值导航课程被压成了一个点**，模型只被告知"末层要同步"，中间层没有任何约束。

课程应有的形状（100m，`n_blocks=12`）：

| 层 i | 0 | 3 | 6 | 9 | 11 |
|---|---|---|---|---|---|
| 物理时间 t | 0.000 | 0.273 | 0.545 | 0.818 | 1.000 |
| `r_target(t)` | **0.034** | 0.180 | 0.575 | 0.893 | **0.966** |

修复：`--sync_curriculum` 改为逐层跟踪。配套改动是 `stats_list` 新增 `order_raw`
字段且**不做 detach** —— 原先所有统计量都 detach 了，逐层 L_sync 拿不到梯度；
其余诊断量（energy/entropy）仍保持 detach，避免诊断量意外参与反传。

默认关（保持与既有基线可比）。

---

## 6. 训练规模与数据量适配（上 1B 前必读）

当前本地数据实测（`datasets/03_dialogue_clean`，60 个 jsonl）：

| 指标 | 值 |
|---|---|
| 文件大小 | 250 MB |
| 样本条数 | 237,707 |
| 总 token（`max_seq_len=128`） | **28.47 M** |
| 平均样本长度 | 105.8 token |
| 长度分布 | p50=106 / p90=181 / p99=282 / max=2952 |
| `max_len=128` 保留 | 85.7% |
| `max_len=256` 保留 | **99.0%** |

**免费收益**：把 `--max_seq_len` 从 128 提到 256，token 量可直接多拿 **≈15%**
（28.47M → ≈32.9M），几乎无额外截断损失。

### 参数/token 配比（Chinchilla 参考 20 token/参数）

| 预设 | 参数量 | 建议 token | 本地 28.5M 覆盖率 | 差多少 |
|---|---|---|---|---|
| `small` | 16 M | 0.32 B | 8.9% | 11× |
| `100m` | 90 M | 1.81 B | 1.6% | **64×** |
| `300m` | 210 M | 4.20 B | 0.7% | 147× |
| `1b` | 1.1 B | 22.0 B | 0.13% | **772×** |

**结论**：
1. 本地 28.5M token 只够**验证 100M 以内的架构**——这也正是当前阶段的全部目标。
2. 1B 想要不严重过拟合，需要 **~20B token（约 200 GB 原始文本）**量级的数据。
魔塔 UltraData 系列（或任何多源混合语料）不是"可选优化"，而是 1B 的**前置条件**。
3. 在小数据上强行训 1B 的唯一结果是：训练 CE 一路下探、验证 CE 早早反弹，
模型只会背下这 23 万条对话。
4. 参考做法：即便算力受限，用 5–10B token 训 1B（约为 Chinchilla 的 1/4）也比
用 0.03B token 训 1B 现实得多。

---

## 7. 损失组成（v0.1 §7）

```
L = L_ce + w_fm·L_fm + w_sync·L_sync + w_drift·L_drift + w_free·L_F
```

| 项 | 作用 | 默认权重 |
|---|---|---|
| `L_ce` | 语言建模交叉熵（末步反嵌入） | 1.0 |
| `L_fm` | ELF 流匹配速度场（去噪分支，共享物理核心权重） | 0.1 |
| `L_sync` | 序参量课程跟踪 `r → σ((t−t_c)/τ_c)` | 0.1 |
| `L_drift` | 漂移场一致性 | 0.05（当前恒 0，待接目标嵌入） |
| `L_F` | 自由能正则 `U − T·S`，含熵下限防全同步坍缩 | 0.05 |

训练日志（`logs/train_log.tsv`）同时记录物理诊断：序参量 `r_mean`、
振幅熵 `entropy`、XY 能量 `energy`。

---

## 8. 设计稿（v0.1）落实情况对照

以 `docx/PMNN物理矩阵神经网络通用底层架构.md` 为准，逐条核对实现状态。
❌ 项都已用 grep 验证过确实未接入（不是"可能没实现"）。

| 设计稿条目 | 状态 | 说明 |
|---|---|---|
| §2 复值振子 `z = a·e^{iθ}` | ✅ | a/θ 分离实数表示，`a≥0` 由 softplus 保证 |
| §3.1 Kuramoto **逐对**耦合 `Σ_j W_ij·sin(θ_j−θ_i)` | ⚠️ 本轮修复 | 原实现退化成平均场近似 `r·sin(ψ−θ_i)`，见 §5.4 |
| §3.1 低秩耦合 `W = UVᵀ` | ✅ | `r_rank` 可配（32/64/128） |
| §3.1 κ(t) 同步强度（K 值导航） | ✅ | |
| §3.2 共振-驻波 FNN（Lorentzian 增益 + 驻波调制） | ✅ | |
| §3.3 漂移场 `V = V⁺ − λV⁻`（多温度 softmax） | ✅ | einsum 重写，反向 23× 加速 |
| §3.3 训练期演化 → 推理期逐轮活跃 | ⚠️ 部分 | 训练期逐轮活跃已实现 |
| §3.4 XY 能量 / 振幅熵 / 自由能 `F = U − T·S` | ✅ | 全部 mask 加权 |
| §3.4 **朗之万噪声** `η ~ N(0, 2T·Δt)` | ✅ 已接入 | 实现于 `KuramotoSync.forward` 积分步，**仅训练期注入**；`--langevin` 开启（默认关，待 A/B） |
| §3.5 Heun/RK4 可逆积分 | ⚠️ 显式 Euler | `dt=0.1`，精度够但不省显存 |
| §5 `∇_Z[−U + T·S]` 自由能梯度 | ✅ 已补齐振幅分量 | 相位分量由 KuramotoSync 承接（`∇_θU`=sin 耦合）；振幅分量 `∇_a[−U+T·S]` 新增 `free_energy_amplitude_gradient`，`--free_energy_amp` 开启（默认关，待 A/B） |
| §6 **128 维轨迹瓶颈** | ✅ 已接入 | 速度场改为由瓶颈预测，`bottleneck` 参数梯度 0.212（修复前为 None） |
| §6 **mode 门控** `g(t)=σ(W_g·[t, r(t), κ(t)])`，`t_dec≈0.98` | ✅ 已实现 | `--mode_gate` 开启，按热力学状态加权 L_ce 与 L_fm（默认关，待 A/B） |
| §6 ELF 流匹配 `x_t = t·x₁ + (1−t)x₀` | ✅ | 整流流 + 速度场目标 |
| §6 共享权重（物理核心兼作速度场网络） | ✅ | `flow_matching` 复用 `blocks[0].sync` |
| §7 `L = L_fm + λ₁L_sync + λ₂L_drift + λ₃L_F` | ✅ + `L_ce` | 文本任务额外加语言建模交叉熵 |
| §7 `L_sync` 序参量课程 `r(t)` 逐层跟踪 | ⚠️ → ✅ | 原实现只约束末层、目标恒为常数 0.9656；`--sync_curriculum` 恢复逐层曲线 |
| §7 K 课程 `κ(t) = κ_max·σ((t−t_c)/τ)` | ✅ | |
| §7 温度退火 `T(t) = T_max(1−t)^p + T_min` | ✅ | |
| §7 两阶段训练（A 预训练核心 / B 端到端流匹配） | ❌ 未分阶段 | 当前为单阶段联合训练 |
| §8 推理：ODE 正向积分 + 漂移场逐轮活跃 | ⚠️ 偏离 | 当前用「物理核心编码 + 自回归解码」保文本可控性 |

**剩余缺口**（按优先级）：
1. 两阶段训练（§7）—— 阶段 A 预训练物理核心 / 阶段 B 端到端流匹配，当前为单阶段联合
2. Heun/RK4 积分（§3.5）—— 当前为显式 Euler（`dt=0.1`），可逆省显存的能力未落地

---

## 9. 常见问题

**Q: 显存不够？**
减小 `--batch`，或用 `--optimizer adafactor`，或降 `--max_seq_len`。

**Q: 数据目录找不到？**
`--data_dir` 指定，或设 `PMNN_DATA_DIR`。脚本会自动向上层查找 `datasets` 目录。

**Q: 改了 `--max_seq_len` 但没生效？**
预处理产物的 `max_len` 小于需求时会自动重新预处理；也可 `--rebuild_prepared` 强制。

**Q: 想换词表大小？**
`prepare_data.py --vocab_size 32768 --rebuild_tokenizer`，然后重新预处理。
模型的 `vocab_size` 会自动对齐词表实际大小。
