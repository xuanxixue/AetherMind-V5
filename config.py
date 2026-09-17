"""PMNN 文本生成模型配置（物理矩阵神经网络，基于 v0.1 架构稿）。

超参集中在此，训练/生成/预处理脚本共用。
路径不在这里 —— 统一由 paths.py 解析（CLI > 环境变量 > 自动探测）。

规模预设（PRESETS）：
    tiny    ~9M    冒烟测试 / 快速验证代码通路
    small   ~18M   本地默认（RTX 3050 4GB 轻松跑）
    100m    ~97M   本地架构验证上限 / 云端小规模基线
    300m    ~208M  云端中等规模
    1b      ~1.27B 历史预设（d=2048 宽 × 24 轮；@8k 词表 1.115B）
    1b32    ~1.00B 目标 1.0B（配 32k 词表）
    1_5b32  ~1.53B 目标 1.5B（配 32k 词表）

云端 1B 用法（不改代码）：
    python train.py --preset 1b32 --tokenizer data/tokenizer_32k.json \
        --steps 300000 --batch 8 --grad_accum 16

维度临时试算（不加预设、不动其它超参）：
    python scripts/param_report.py --preset 1b32 --d_model 2048 --n_blocks 28
"""
from dataclasses import dataclass
from typing import Dict


# ----------------------------------------------------------------------------
# 规模预设：只覆盖「结构维度」，其余超参保持默认
#
# ⚠️ 参数量**依赖词表规模**（token_embed + lm_head = V×(d_model + 2·d_field)）。
#    d=2048/V=32768 时词表项就是 201M —— 占 1.25B 模型的 16%。
#    所以下面的 "1b"/"1_5b" 标注的是**该预设配 32k 词表时**的参数量；
#    换成 8k 词表会各少约 151M。用 scripts/param_report.py 可随时核对
#    （它用 meta 设备真实实例化模型来校验解析式，不是估算）。
# ----------------------------------------------------------------------------
PRESETS: Dict[str, dict] = {
    # 名称        d_model d_field n_blocks ffn_hidden r_rank
    "tiny":  dict(d_model=256,  d_field=256,  n_blocks=4,  ffn_hidden=768,   r_rank=16),
    "small": dict(d_model=384,  d_field=384,  n_blocks=6,  ffn_hidden=1024,  r_rank=32),
    "100m":  dict(d_model=768,  d_field=768,  n_blocks=12, ffn_hidden=3072,  r_rank=64),
    "300m":  dict(d_model=1024, d_field=1024, n_blocks=16, ffn_hidden=4096,  r_rank=96),
    # 历史预设：d=2048 宽 + 24 轮。@8k 词表 1.115B，@32k 词表 1.266B。
    "1b":    dict(d_model=2048, d_field=2048, n_blocks=24, ffn_hidden=8192,  r_rank=128),
    # ↓ 按目标参数量定制的两档（词表按推荐的 32768 计）
    # 1b32  ≈1.000B @32k：守 24 轮物理时间（相变区 t∈[0.35,0.65] 有 ~7 轮采样），
    #   d 收到 1792、ffn 收到 7168，把预算让给词表和耦合秩。
    "1b32":  dict(d_model=1792, d_field=1792, n_blocks=24, ffn_hidden=7168, r_rank=160),
    # 1_5b32 ≈1.533B @32k：同族放大（ffn/d 仍 4×、耦合秩仍 160），
    #   宽度 ×1.14、轮数 ×1.25 —— 轮数涨得比宽度快，因为相变区分辨率只由轮数决定。
    "1_5b32": dict(d_model=2048, d_field=2048, n_blocks=30, ffn_hidden=8192, r_rank=160),
}


def scale_dims(cfg, **dims) -> "PMNNConfig":
    """直接覆盖结构维度（不改预设名、不动其它超参）。

    用途：`--d_model/--n_blocks/...` 这类临时试验，避免为了试一个维度
    就去 config.py 里加一个预设。会同步刷新由维度派生的量（n_drive_freq）。

        cfg = scale_dims(get_config("300m"), d_model=1536, n_blocks=20)
    """
    for k, v in dims.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"未知结构维度: {k}")
        setattr(cfg, k, int(v))
    cfg.n_drive_freq = max(32, min(128, cfg.d_field // 8))
    return cfg


@dataclass
class PMNNConfig:
    # ---- 规模预设名（记录用，训练/生成一致）----
    preset: str = "small"

    # ---- 词表 / 序列 ----
    vocab_size: int = 8192          # 由词表文件实际大小覆盖
    max_seq_len: int = 128          # 训练序列长度（4GB 显存约束）
    pad_token_id: int = 0

    # ---- 场维度（复值振子状态 z = a · e^{iθ}）----
    d_model: int = 384              # 输入嵌入维度（振子化前）
    d_field: int = 384              # 正子场通道数（振幅 a / 相位 θ 均为 N×d_field）
    # ELF 轨迹瓶颈维度（设计稿 §6：512 维 T5 嵌入 → 128 维瓶颈）
    #   速度场从瓶颈预测，而不是直接从高维场预测；这是 ELF 的核心技巧。
    bottleneck_dim: int = 128
    # mode 门控（设计稿 §6）：g(t) = σ(W_g·[t, r(t), κ(t)])，t_dec ≈ 0.98
    #   去噪分支（t < t_dec）与解码分支（t ≥ t_dec）的软切换。
    #   本工程落地为「按同步状态动态加权 L_ce 与 L_fm」——设计稿 §4 的
    #   「自适应加权：早期 L_phys 主导 → 后期 L_align 主导」。
    #   默认关：会改变两条分支的权重配比，需 A/B 对照确认收益。
    mode_gate: bool = False
    t_dec: float = 0.98             # 解码分支切换阈值（记录用，见 mode_gate 说明）

    # ---- 物理核心 ----
    n_blocks: int = 6               # R 轮物理块（对位：Transformer 层数）
    r_rank: int = 32                # 低秩耦合矩阵 W = U·Vᵀ 的秩
    dt: float = 0.1                 # Kuramoto 每块积分步长
    # 因果距离衰减耦合（替代注意力的因果路由，语言建模必需）
    #   C_i = Σ_{j≤i} e^{−λ(i−j)}·sin(θ_j − θ_i)，λ = ln2 / causal_halflife
    #   用 cumsum 实现 O(N·D)，比 attention 的 O(N²) 更省且天然因果
    causal_coupling: bool = True
    causal_halflife: float = 16.0   # 衰减半衰期（token），越小越"近视"
    causal_mix: float = 0.7         # 因果项混合：0=纯全局(置换不变)，1=纯因果
    # 跨位置耦合方式 —— 决定模型能否「按内容检索」上下文（这是语言建模的关键能力）
    #   mean_field: 低秩聚合 + 固定距离衰减（原实现）。聚合权重与「自己的内容」无关，
    #               只能形成上下文的 rank-1 摘要，模型不具备内容寻址能力。
    #   content:    内容驱动的因果耦合（物理版注意力）。耦合强度由振子状态本身决定：
    #                   W_ij ∝ exp( q_i·k_j / √r − λ·(i−j) ),  i ≥ j
    #               相位对齐、振幅大的振子之间耦合更强 —— 这正是「同步」的物理含义，
    #               也是 attention 在物理框架下的对应物。
    coupling: str = "content"
    coupling_halflife: float = 64.0  # content 耦合的距离衰减半衰期（比 mean_field 更长程）
    coupling_temp: float = 1.0       # 耦合锐度（softmax 温度，越小越"精准检索"）

    # ---- 共振-驻波 FNN ----
    ffn_hidden: int = 1024          # FNN 隐藏维
    n_wave_modes: int = 16          # 驻波模式数 k_m = 2πm/N

    # ---- 漂移场 ----
    drift_tau: float = 1.0          # 多温度 softmax 主温度
    drift_lambda: float = 0.5       # 排斥项权重 λ（V = V⁺ − λV⁻）
    n_temps: int = 2                # 多尺度温度数 τ ∈ {τ, 2τ}
    n_proto: int = 32               # 吸引原型数（数据分布吸引中心）

    # ---- 热力学 / 时间场 ----
    kappa_max: float = 8.0          # 耦合强度上限（K 值导航）
    t_c: float = 0.5                # 同步相变中心
    tau_c: float = 0.15             # 相变锐度
    T_max: float = 1.0              # 温度退火起点
    T_min: float = 0.05             # 温度退火终点
    temp_power: float = 1.5         # 退火曲线指数
    n_drive_freq: int = 32          # 驱动器/固有频率通道
    # 朗之万噪声（设计稿 §3.4 涨落-耗散 / §5 状态方程的 η(t) 项）
    #   θ += √(2T·dt)·N(0,1)，只在训练期注入；T(t) 退火到 T_min 后噪声自然趋 0
    #   默认关：它是训练期正则/探索项，需 A/B 对照确认收益后再开。
    langevin: bool = False
    langevin_gain: float = 1.0      # 噪声幅度系数（1.0 = 严格按涨落-耗散定理）

    # 自由能振幅梯度（设计稿 §5 状态方程的振幅分量 ∇_a[−U+T·S]）
    #   此前振幅只靠 FNN 残差更新，缺物理势能驱动力；此开关把自由能下降方向
    #   作为加性驱动项并入振幅更新（相位分量已由 KuramotoSync 承担）。
    #   默认关：会改变振幅演化动力学，需 A/B 对照确认收益后再开。
    free_energy_amp: bool = False
    free_energy_amp_gain: float = 1.0  # 振幅梯度缩放（1.0 = 状态方程原始量纲）

    # ---- 损失权重（v0.1 §7）----
    w_fm: float = 0.1               # L_fm   ELF 流匹配
    w_sync: float = 0.1             # L_sync 序参量课程跟踪
    # 序参量课程是否**逐层**跟踪（设计稿 §7 的 r(t) 课程）
    #   r_target(t) = σ((t − t_c)/τ_c)，t = 层深度 i/(n_blocks−1)
    #   第 0 层 r≈0.034（混沌探索）→ 末层 r≈0.966（同步整合）
    #   False = 旧行为：只约束末层，目标恒为常数 0.9656 —— 整个课程结构丢失。
    sync_curriculum: bool = False
    w_drift: float = 0.05           # L_drift 漂移场一致性
    w_free: float = 0.05            # L_F    自由能正则
    entropy_target: float = 1.0     # 熵压目标（防全同步坍缩）
    w_entropy_floor: float = 0.1    # 熵下限权重
    label_smoothing: float = 0.0

    # ---- 训练 ----
    batch_size: int = 16
    grad_accum: int = 2             # 有效 batch = batch_size × grad_accum
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    max_steps: int = 6000
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True            # 混合精度（bf16）
    seed: int = 42
    num_workers: int = 0            # Windows 下 0 最稳（避免 spawn 复制内存）
    optimizer: str = "adamw"        # adamw | adafactor（省显存）
    dynamic_pad: bool = False       # True=按 batch 动态补齐；False=固定 max_seq_len
    max_samples: int = 0            # 0 = 用全部预处理样本
    eval_every: int = 500
    save_every: int = 500           # 0 = 只在结束时保存。
    # 攒太长会丢进度：间隔 5000 步时中途一断，最后 5000 步全白跑（实测踩过）。
    # 500 步 + keep_ckpt=5 是「掉电最多损失 500 步、磁盘只留 5 份」的平衡点。
    # 另有中断兜底存档（train.py 的 SIGINT/SIGTERM 处理），不依赖本间隔。
    # checkpoint 是否携带优化器动量。续训必需（不存的话 AdamW 动量从零重建，
    # 续训初期 loss 会反弹）；纯推理/交付用的 checkpoint 可以关掉，省约 3/4 空间
    # （100m: 1.2GB → 0.36GB；1B: ≈13GB → ≈4GB）。
    save_optimizer: bool = True
    keep_ckpt: int = 5              # 只保留最新 N 个 checkpoint，旧的自动删除（0=不删）
    log_every: int = 20
    # 收敛早停：CE 滑窗均值连续 patience 步不再刷新最优（改善幅度需 > min_delta）即自动停止。
    # CE 天然在 1~3 之间抖动，所以用「滑窗均值」判断而非单点，避免误触发。
    # patience=0 关闭。云端建议 3000（≈25 分钟 @2 step/s），不必训满 max_steps。
    early_stop_patience: int = 0    # 0 = 关闭
    early_stop_window: int = 5      # 滑窗长度（单位=日志记录点，×log_every 步）
    early_stop_min_delta: float = 0.01
    # ---- wall-clock 软退出（云端「任务启动即计时」平台必设）----
    # 平台单次任务通常限时（如 10 小时），到点直接强杀进程 —— 那样最后一个
    # 存档间隔（默认 500 步）的训练成果全丢，白烧机时。此参数让训练自己
    # 在到点前主动存盘并正常退出，配合 --resume 即可无缝接续下一轮。
    #   0 = 不限制；云端建议 9.0（10h 任务留 1h 余量给存盘 + 排队 + 上传）
    max_hours: float = 0.0
    # 到点前多少分钟进入收尾。1B 完整存档约 15GB，云端网络盘写入慢，
    # 留 20 分钟是「存得完」与「不浪费机时」的平衡点。
    wallclock_margin_min: float = 20.0
    resume: str = ""                # checkpoint 路径，续训
    # 防呆：checkpoints/ 里已有存档、却又没传 --resume 时，train.py 直接中止。
    # 这是最贵的一类错误（白烧几小时机时 + 覆盖旧档），必须显式确认才能从头训。
    fresh_ok: bool = False          # True = 明知有存档，但确实要重训
    # 防呆：同一份 checkpoints/ 目录被多个训练进程同时写入时，
    # 会互相覆盖存档、白烧机时。默认硬拒绝，要并行必须显式声明。
    force_start: bool = False       # True = 无视单实例锁，强行启动
    dry_run: bool = False           # 只测显存/前向反向，不入库

    # ---- 生成 ----
    gen_temperature: float = 0.8
    gen_top_k: int = 50
    gen_top_p: float = 0.9
    gen_max_new_tokens: int = 64


def get_config(preset: str = None, **overrides) -> PMNNConfig:
    """按预设 + 覆盖项构建配置。"""
    cfg = PMNNConfig()
    if preset:
        apply_preset(cfg, preset)
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"未知配置项: {k}")
        setattr(cfg, k, v)
    return cfg


def apply_preset(cfg: PMNNConfig, name: str) -> PMNNConfig:
    """把规模预设写入配置。"""
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name}，可选: {list(PRESETS)}")
    cfg.preset = name
    for k, v in PRESETS[name].items():
        setattr(cfg, k, v)
    # 规模相关的换算
    cfg.n_drive_freq = max(32, min(128, cfg.d_field // 8))
    return cfg


def add_config_args(parser) -> None:
    """给 argparse 挂上模型/训练超参（与 dataclass 字段一一对应）。"""
    g = parser.add_argument_group("规模与超参")
    g.add_argument("--preset", default="small", choices=list(PRESETS),
                   help="规模预设（默认 small≈16M）。1b32≈1.0B / 1_5b32≈1.5B "
                        "（按 32k 词表计）")
    # ---- 结构维度覆盖（不改预设名，试维度用）----
    g.add_argument("--d_model", type=int, default=None, help="覆盖嵌入维度")
    g.add_argument("--d_field", type=int, default=None, help="覆盖振子场通道数")
    g.add_argument("--n_blocks", type=int, default=None, help="覆盖物理块轮数")
    g.add_argument("--ffn_hidden", type=int, default=None, help="覆盖 FNN 隐藏维")
    g.add_argument("--r_rank", type=int, default=None, help="覆盖耦合低秩")
    g.add_argument("--n_wave_modes", type=int, default=None, help="覆盖驻波模式数")
    g.add_argument("--steps", type=int, default=None, dest="max_steps")
    g.add_argument("--batch", type=int, default=None, dest="batch_size")
    g.add_argument("--grad_accum", type=int, default=None)
    g.add_argument("--lr", type=float, default=None, dest="learning_rate")
    g.add_argument("--max_seq_len", type=int, default=None)
    g.add_argument("--warmup_steps", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--num_workers", type=int, default=None)
    g.add_argument("--max_samples", type=int, default=None,
                   help="限制使用的预处理样本数（0=全部）")
    g.add_argument("--optimizer", default=None, choices=["adamw", "adafactor"])
    g.add_argument("--no_amp", action="store_true", help="关闭混合精度")
    g.add_argument("--no_causal", action="store_true",
                   help="关闭因果衰减耦合（用于 A/B 对照实验）")
    g.add_argument("--causal_halflife", type=float, default=None,
                   help="因果耦合衰减半衰期（token）")
    g.add_argument("--causal_mix", type=float, default=None,
                   help="因果项混合系数 0~1")
    g.add_argument("--coupling", default=None, choices=["mean_field", "content"],
                   help="跨位置耦合方式：mean_field=低秩固定衰减 / content=内容寻址因果耦合")
    g.add_argument("--coupling_halflife", type=float, default=None,
                   help="content 耦合的距离衰减半衰期（token）")
    g.add_argument("--coupling_temp", type=float, default=None,
                   help="content 耦合的 softmax 锐度")
    g.add_argument("--langevin", action="store_true",
                   help="训练期注入朗之万噪声（涨落-耗散，设计稿 §3.4）")
    g.add_argument("--langevin_gain", type=float, default=None,
                   help="朗之万噪声幅度系数（1.0 = 严格涨落-耗散）")
    g.add_argument("--free_energy_amp", action="store_true",
                   help="把自由能振幅梯度并入振幅更新（设计稿 §5 振幅分量）")
    g.add_argument("--free_energy_amp_gain", type=float, default=None,
                   help="自由能振幅梯度缩放系数（默认 1.0）")
    g.add_argument("--mode_gate", action="store_true",
                   help="开启 mode 门控（按 r/κ 动态加权 L_ce 与 L_fm，设计稿 §6）")
    g.add_argument("--t_dec", type=float, default=None,
                   help="解码分支切换阈值（默认 0.98）")
    g.add_argument("--sync_curriculum", action="store_true",
                   help="L_sync 按层跟踪序参量课程 r(t)（设计稿 §7）")
    g.add_argument("--dynamic_pad", action="store_true", help="按 batch 动态补齐序列")
    g.add_argument("--eval_every", type=int, default=None)
    g.add_argument("--save_every", type=int, default=None,
                   help="每 N 步存一次 checkpoint（0=只在结束存）")
    g.add_argument("--keep_ckpt", type=int, default=None,
                   help="只保留最新 N 个 checkpoint，旧的自动删除（0=不删）")
    g.add_argument("--no_save_optimizer", action="store_true",
                   help="checkpoint 不保存优化器状态（省约 3/4 空间，但无法续训）")
    g.add_argument("--log_every", type=int, default=None)
    g.add_argument("--early_stop_patience", type=int, default=None,
                   help="CE 滑窗均值连续 N 步不刷新最优即早停（0=关闭，云端建议 3000）")
    g.add_argument("--early_stop_window", type=int, default=None,
                   help="早停滑窗长度（日志记录点数，默认 5，×log_every 步）")
    g.add_argument("--early_stop_min_delta", type=float, default=None,
                   help="判定「有改善」的最小 CE 下降幅度（默认 0.01）")
    g.add_argument("--max_hours", type=float, default=None,
                   help="单次进程 wall-clock 上限（小时），到点前自动存盘退出（0=不限）")
    g.add_argument("--wallclock_margin_min", type=float, default=None,
                   help="到点前多少分钟开始收尾存盘（默认 20）")
    g.add_argument("--resume", default=None, help="从 checkpoint 续训")
    g.add_argument("--fresh_ok", action="store_true",
                   help="确认「从头开始训练」：checkpoints/ 里即使有存档也不中止")
    g.add_argument("--force_start", action="store_true",
                   help="无视单实例锁强行启动（同一存档目录已被别的训练进程占用时用）")
    g.add_argument("--dry_run", action="store_true", help="只测显存与前后向，不训练")


def config_from_args(args) -> PMNNConfig:
    """从 argparse 结果构建配置（未提供的项保持默认/预设）。"""
    cfg = get_config(preset=getattr(args, "preset", None) or "small")
    for field_name in PMNNConfig.__dataclass_fields__:
        cli = getattr(args, field_name, None)
        if cli is not None:
            setattr(cfg, field_name, cli)
    if getattr(args, "no_amp", False):
        cfg.use_amp = False
    if getattr(args, "no_causal", False):
        cfg.causal_coupling = False
    if getattr(args, "no_save_optimizer", False):
        cfg.save_optimizer = False
    # d_field 可能被 CLI 覆盖 → 由维度派生的量必须重算，
    # 否则 n_drive_freq 会停留在预设值（维度变了却不一致）。
    cfg.n_drive_freq = max(32, min(128, cfg.d_field // 8))
    return cfg
