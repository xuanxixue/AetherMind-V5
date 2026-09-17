# ModelScope 云端训练操作手册（PMNN 300M）

> 目标：把本包上传到 ModelScope 免费 GPU 实例，解压后一条命令满速训练 300M。
> 环境：8 核 32GB CPU + 24GB 显存，36 小时（启动即计时，单次运行最长 10 小时，空闲 1 小时自动关停）。

## 准备工作（本机，已完成）
- `pmnn_cloud_300m.tar.gz`（约 31MB）：内含代码 + 预训练数据：
  - `data/tokenizer.json` —— 词表（vocab=8192）
  - `prepared/` —— 预分词产物 tokens.bin + index.npy + meta.json（128 长度 × 237707 样本 × 2847 万 token）
  - 脚本与代码，含 `scripts/run_cloud_300m.sh`
- `pmnn_guard_patch.tar.gz`（约 25KB）：**断点保护 + 续训自检补丁**
  （train.py / config.py / scripts/run_cloud_300m.sh / CLOUD_GUIDE.md 四个文件），覆盖解压即可获得：
  - 每 500 步存档 + 只留最新 5 份 + Ctrl+C 紧急存档 + 崩溃兜底 + 收敛早停；
  - **续训自检**：启动就打印「是否续训 / 从第几步起 / 权重是否真的加载进去」，续训档不存在直接中止；
  - **`RESUME=auto`**：自动挑最新存档，告别手写 sed；
  - **「有存档却不续训」硬拦截**：`checkpoints/` 里已有档却没传 `--resume` 时 *train.py 直接中止*，
    不再静默从 0 重训（确认要重训才加 `--fresh_ok`）；脚本会自动带上 `--fresh_ok`，正常流程无感；
  - python/python3 自动探测（镜像只注册 python3 也不会报错退出）；
  - 脚本开头会打印 `补丁版本: guard+resume-check v3 ...` —— **这一行就是补丁有没有生效的唯一凭证**。

> **解压补丁务必在 `/mnt/workspace` 下执行**（包内路径是 `pmnn_text/...`）。
> 若在 `/mnt/workspace/pmnn_text` 里解压，会多出一层 `pmnn_text/pmnn_text/`，
> 旧脚本原封不动 → 就是「补丁打了等于没打」。

## 云端操作步骤（照顺序）

### 第 1 步：创建 GPU 实例
1. 打开 https://www.modelscope.cn/my/mynotebook
2. 选 **GPU 免费资源**（8 核 32GB + 24GB 显存），创建内核，**先选好镜像**（推荐 PAI-DSW 默认 / 或带 `pytorch` 的镜像，torch 已预装）
3. 进入 JupyterLab / Terminal

### 第 2 步：上传压缩包
- 打开左侧文件树，把本机压缩包拖进实例工作区（当前实例是 **`/mnt/workspace`**；`/workspace` 不存在）

### 第 3 步：解压
```bash
cd /mnt/workspace
tar -xzf pmnn_cloud_300m.tar.gz
cd pmnn_text
```
若还要打补丁（老包才需要；新包已内置）：
```bash
cd /mnt/workspace && tar -xzf pmnn_guard_patch.tar.gz && cd pmnn_text
```

### 第 4 步：（可选）摸底显存
```bash
python train.py --preset 300m --dry_run --batch 32 --grad_accum 4 --optimizer adamw --coupling content
```
- 实测峰值 **11.7GB / 22.2GB（53%）**，24GB 对 300M 极其宽裕。

### 第 5 步：后台启动训练（防止关掉网页终端被杀）
```bash
nohup bash scripts/run_cloud_300m.sh > train_300m.log 2>&1 &
tail -f train_300m.log        # 看着它跑，Ctrl+C 只是退出 tail，不影响训练
```
- 默认：`STEPS=30000`（**总目标步数**，不是剩余）、有效 batch 128、4 进程、adamw、content 耦合、早停 3000。
- 覆盖任意参数：`STEPS=40000 EARLY_STOP=6000 SAVE_EVERY=500 KEEP_CKPT=5 bash scripts/run_cloud_300m.sh`
- 启动时会打印一行 `>>> 总目标 = X 步（已跑 Y 步），本轮实际训练 Z 步`，照它对表即可，别再靠猜。

### 第 6 步：查看进度 / 回收产物
- 训练日志：`logs/train_log_300m.tsv`（CE / r / entropy / lr 每 20 步一条）
- 检查点：`checkpoints/` —— **每 500 步一存，只自动保留最新 5 份**（旧的自动删，不会撑爆磁盘）
- 跑完后**务必下载** `checkpoints/` 与 `logs/` 回本地（36 小时到点，实例会被回收，数据会丢！）

## 收敛早停（默认开启，不必训满 —— CE 会在 1~3 之间抖动）

`--steps` 只是**上限**。默认开启早停后，一旦 CE 长时间不再变好就自动收工、存档、退出：

| 参数 | 默认 | 含义 |
|---|---|---|
| `EARLY_STOP` | `3000` | CE 滑窗均值连续 3000 步不刷新最优即停止（≈25 分钟 @2 step/s）；**0 = 关闭** |
| `EARLY_WINDOW` | `5` | 滑窗长度（日志记录点数，×20 步 = 100 步均值） |
| `EARLY_DELTA` | `0.01` | 判定「有改善」的最小 CE 降幅 |

- **为什么用滑窗均值**：CE 单点在 1~3 之间跳动是常态，看单点会被噪声骗；滑窗均值才代表真实趋势。
- 触发时会打印 `[train] 触发收敛早停：... 历史最优 X @ step Y`，存档名带 `earlystop` 标记。
- 想训得更久：`EARLY_STOP=6000 bash scripts/run_cloud_300m.sh`；想跑满不看趋势：`EARLY_STOP=0`。

## 断点保护（不会再出现「跑到 9000 步只存到 5000」）

| 机制 | 行为 |
|---|---|
| 节奏存档 | 每 **500 步**落盘一次 → 意外最多损失 500 步（约 3 分钟） |
| 自动轮转 | 只保留最新 **5 份**，旧的自动删除 → 磁盘占用恒定 |
| Ctrl+C | 捕获 SIGINT/SIGTERM → **本步结束后立即存档再退出**，不丢进度 |
| 手抖连按 | 3 秒内连按视为同一次，**不会**跳过存档；等 3 秒后再按才会放弃存档强制退出 |
| 崩溃兜底 | 任何异常（CUDA OOM 等）都会在退出前把最后完成的步数存下来 |
| 原子写盘 | 先写 `.pt.tmp` 再改名 → 绝不会留下「名字正常、内容写坏」的坏档 |

### 万一中断了，怎么续训
```bash
cd /mnt/workspace/pmnn_text
# 一句话续训：RESUME=auto 自动挑 step 最大的那个存档（不用再手写 sed/管道）
STEPS=30000 RESUME=auto nohup bash scripts/run_cloud_300m.sh > train_300m_resume.log 2>&1 &
sleep 60; head -40 train_300m_resume.log      # 看开头 40 行，重点看续训自检
```
> **最容易搞混的一点**：`--steps` / `STEPS` 是**总目标步数**（含已跑部分），**不是「还要再跑多少步」**。
> 例：已到 9500 步、想总共跑到 30000 步 → `STEPS=30000`（本轮实际再跑 20500 步）。
> 若写 `STEPS=15000` 就会在 15000 步停下（只再跑 5500 步）。lr 按绝对步数接着原 cosine 曲线走，不会重新 warmup。

### 怎么确认「续训真的生效了」（别再靠猜）
起训后日志开头必须有这几行，缺任何一行就是没接上：

```
[cloud-300m] 补丁版本: guard+resume-check v3 (2026-09-13)   <- 没有这行 = 补丁没生效，停下来先打补丁
[cloud-300m] RESUME=auto -> 自动选中最新档: checkpoints/pmnn_300m_step5000.pt
[cloud-300m] >>> 总目标 = 30000 步（已跑 5000 步），本轮实际训练 25000 步
[cloud-300m] >>> 实际执行命令（可直接复制手动跑）：
    python3 -u train.py ... --resume checkpoints/pmnn_300m_step5000.pt
[train] 命令行: ... --resume checkpoints/pmnn_300m_step5000.pt
[train] 已加载续训权重: ... (存档 step=5000)
[train] 权重键完全匹配（N 项全部命中）
[train] >>> 续训模式：global_step 从 5000 起（本轮再跑 25000 步）
[train] 起步自检：固定 batch loss=1.7xxx （随机初始化参考 ≈ ln(vocab) = 9.01）
```

判定标准（一句话版）：
- **自检 loss ≈ 9.0** → 权重没生效（等于从零开始），要停下来查；
- **自检 loss 明显低于 9.0** → 接上了，放心跑；
- 第一个进度行应该是 `[step 50xx/30000]`；**若看到 `[step 1/30000]` 或 `[step 20/30000]` 就是没接上**。

### ⚠️ 症状对照表：「明明有存档，它却从 0 开始训」
这是最容易白烧机时的一类故障。**只看日志就能定性**：

| 日志里看到 | 含义 | 处理 |
|---|---|---|
| `[cloud-300m] workdir` 后**直接**是 `preset=...`（没有 `补丁版本: ... v3` 行） | 跑的是**旧脚本**，`RESUME=auto` 会被静默丢掉 | 在 `/mnt/workspace` 重新解压补丁包 |
| `>>> 实际执行命令` 那一行里**没有 `--resume`** | 确实没传续训参数 | 同上 |
| `python train.py ... --log_every 20`（没有 `--keep_ckpt` / `--early_stop_*`） | 旧脚本的痕迹 | 同上 |
| `[train] >>> 全新训练：global_step 从 0 起（本次未使用 --resume）` | 已经在从头训了 | `pkill -f train.py` 立刻停掉 |
| `[train] 错误：... 下已有 N 个存档，但本次**没有**传 --resume` | 新版防呆拦住了 | 这是**好事**：加 `--resume`（或用 `RESUME=auto`）重跑 |

> 为什么必须立刻停：从头训的那个进程跑到第 5000 步时，会写出**同名**的
> `pmnn_300m_step5000.pt`，把你原来那个好存档覆盖掉 —— 那才是真正的损失。

### ⚠️ 绝对不要这么干（本次踩的坑）
```bash
# ❌ 直接裸跑 train.py：绕开脚本 ⇒ 没有 --resume、没有早停、没有断点保护
python train.py --preset 300m --steps 30000 > train_300m.log 2>&1 &
# ❌ 两个进程写同一个日志文件：`>` 会截断，把前一次的记录冲掉，事后无从对证
nohup bash scripts/run_cloud_300m.sh > train_300m.log 2>&1 &
```
- 想要「总目标步数」，用 `STEPS=`（大写，环境变量）；`--steps` 是脚本内部才用的参数。
- 每次重启**换一个日志文件名**（`train_300m_resume.log`），历史记录留着才查得出问题。
- 续训**只需一条命令**（上面的 `RESUME=auto` 那条），别自己拼 `python train.py`。


## 监控要点（防白烧 36 小时）
- 首 100 步：确认 CE 从 ~9 往下降、lr 正常、无报错。
- **步速别看第 1 步**（含 CUDA 预热，显示 0.3~0.4 step/s），跑到几十步后会飙到 **2+ step/s**。
- 实测：8000~9000 步约 1 小时 → 15000 步 ≈ 1.5 小时，30000 步 ≈ 3.5 小时，10 小时上限非常宽裕。
- run 脚本用 `set -euo pipefail`，任何一步失败会立刻停止报错，不会空转烧时长。

## 关键说明（为什么这么设计的）
1. **不用重跑数据清洗/预分词**：包内已带 `prepared/` 预分词产物，`train.py` 会自动复用（无 jsonl 也能直接起训）。
2. **seq_len=128**：与预分词产物严格一致，不要改成 256（会触发重跑预处理 + 更长序列，白烧时间）。
3. **content 耦合**：本地 100M 实证 −1.5%；1-epoch 摸到 CE 1.865（修正 lr 周期后，之前 4.2 是人为截断）。
4. **36 小时是「启动即计时」**：上传前把包备好、命令准备好，上去就解压+起训，中间不要挂空。
