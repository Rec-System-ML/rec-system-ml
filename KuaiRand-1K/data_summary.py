# -*- coding: utf-8 -*-
"""
KuaiRand-1K 数据集概览分析脚本
- 小文件：完整读取
- 大文件 (log_standard, video_features_basic)：读取前 5 万行采样
- 超大文件 (video_features_statistic, 3.2GB)：分块统计，不全量加载
"""

import pandas as pd
import numpy as np
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SEP = "=" * 65


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def basic_info(df, name, sampled=False):
    note = " [前5万行采样]" if sampled else ""
    print(f"\n【{name}】{note}")
    print(f"  行数: {len(df):,}  | 列数: {df.shape[1]}")
    print(f"  列名: {list(df.columns)}")
    print(f"  数据类型:\n{df.dtypes.to_string()}")
    print(f"\n  缺失值统计:")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("    无缺失值")
    else:
        print(miss.to_string())
    print(f"\n  数值列统计摘要:")
    print(df.describe(include=[np.number]).T[["min", "mean", "max", "count"]].to_string())


# ──────────────────────────────────────────────
# 1. 用户特征（小文件，完整读取）
# ──────────────────────────────────────────────
section("1. 用户特征  user_features_1k.csv  (~128KB)")

user = pd.read_csv(os.path.join(DATA_DIR, "user_features_1k.csv"))
basic_info(user, "user_features")

print(f"\n  用户数: {user['user_id'].nunique():,}")
print(f"  活跃度分布:\n{user['user_active_degree'].value_counts().to_string()}")
print(f"  是否主播: {dict(user['is_live_streamer'].value_counts())}")
print(f"  是否创作者: {dict(user['is_video_author'].value_counts())}")
print(f"  注册天数范围:\n{user['register_days_range'].value_counts().to_string()}")
print(f"  粉丝数范围:\n{user['fans_user_num_range'].value_counts().to_string()}")


# ──────────────────────────────────────────────
# 2. 随机曝光日志（小文件，完整读取）
# ──────────────────────────────────────────────
section("2. 随机曝光日志  log_random_4_22_to_5_08_1k.csv  (~3MB)")

log_rand = pd.read_csv(os.path.join(DATA_DIR, "log_random_4_22_to_5_08_1k.csv"))
basic_info(log_rand, "log_random")

print(f"\n  用户数: {log_rand['user_id'].nunique():,}")
print(f"  视频数: {log_rand['video_id'].nunique():,}")
print(f"  时间跨度: {log_rand['date'].min()} → {log_rand['date'].max()}")

feedback_cols = ["is_click", "is_like", "is_follow", "is_comment",
                 "is_forward", "is_hate", "long_view"]
print(f"\n  反馈信号正例率:")
for col in feedback_cols:
    rate = log_rand[col].mean() * 100
    print(f"    {col:<20}: {rate:.2f}%")

print(f"\n  is_rand=1 (随机干预) 占比: {log_rand['is_rand'].mean()*100:.2f}%")
print(f"  Tab场景分布:\n{log_rand['tab'].value_counts().sort_index().to_string()}")


# ──────────────────────────────────────────────
# 3. 标准日志 Phase1（大文件，采样5万行）
# ──────────────────────────────────────────────
section("3. 标准日志 Phase1  log_standard_4_08_to_4_21_1k.csv  (~357MB)")

log1 = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_1k.csv"), nrows=50_000)
basic_info(log1, "log_standard_phase1", sampled=True)

print(f"\n  [采样] 用户数: {log1['user_id'].nunique():,}")
print(f"  [采样] 视频数: {log1['video_id'].nunique():,}")
print(f"  时间跨度: {log1['date'].min()} → {log1['date'].max()}")
feedback_cols_std = ["is_click", "is_like", "is_follow", "is_comment",
                     "is_forward", "is_hate", "long_view"]
print(f"\n  反馈信号正例率 [采样]:")
for col in feedback_cols_std:
    rate = log1[col].mean() * 100
    print(f"    {col:<20}: {rate:.2f}%")

# 完整行数（用 wc 等价的 chunk 计数）
print(f"\n  完整行数统计（分块扫描）...")
t0 = time.time()
total_rows = sum(len(chunk) for chunk in
                 pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_1k.csv"),
                             usecols=["user_id"], chunksize=500_000))
print(f"  完整行数: {total_rows:,}  (耗时 {time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────
# 4. 标准日志 Phase2（大文件，采样5万行）
# ──────────────────────────────────────────────
section("4. 标准日志 Phase2  log_standard_4_22_to_5_08_1k.csv  (~470MB)")

log2 = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_1k.csv"), nrows=50_000)
basic_info(log2, "log_standard_phase2", sampled=True)

print(f"\n  时间跨度: {log2['date'].min()} → {log2['date'].max()}")
print(f"\n  完整行数统计（分块扫描）...")
t0 = time.time()
total_rows2 = sum(len(chunk) for chunk in
                  pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_1k.csv"),
                              usecols=["user_id"], chunksize=500_000))
print(f"  完整行数: {total_rows2:,}  (耗时 {time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────
# 5. 视频基础特征（大文件，采样5万行）
# ──────────────────────────────────────────────
section("5. 视频基础特征  video_features_basic_1k.csv  (~360MB)")

vbasic = pd.read_csv(os.path.join(DATA_DIR, "video_features_basic_1k.csv"), nrows=50_000)
basic_info(vbasic, "video_features_basic", sampled=True)

print(f"\n  视频类型分布:\n{vbasic['video_type'].value_counts().to_string()}")
print(f"  上传类型分布:\n{vbasic['upload_type'].value_counts().to_string()}")
print(f"  可见状态分布: {dict(vbasic['visible_status'].value_counts())}")
print(f"  视频时长(ms) — min:{vbasic['video_duration'].min():.0f}  "
      f"median:{vbasic['video_duration'].median():.0f}  "
      f"max:{vbasic['video_duration'].max():.0f}")
print(f"  Tag 非空率: {vbasic['tag'].notna().mean()*100:.1f}%")
# Tag 多hot 类别数量分布
tag_counts = vbasic['tag'].dropna().apply(lambda x: len(str(x).split(',')))
print(f"  每个视频平均Tag数: {tag_counts.mean():.2f}  最多: {tag_counts.max()}")

print(f"\n  完整视频数（分块扫描）...")
t0 = time.time()
total_videos = sum(len(chunk) for chunk in
                   pd.read_csv(os.path.join(DATA_DIR, "video_features_basic_1k.csv"),
                               usecols=["video_id"], chunksize=200_000))
print(f"  完整行数: {total_videos:,}  (耗时 {time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────
# 6. 视频统计特征（超大文件 3.2GB，分块读取）
# ──────────────────────────────────────────────
section("6. 视频统计特征  video_features_statistic_1k.csv  (~3.2GB)  [分块扫描]")

stat_path = os.path.join(DATA_DIR, "video_features_statistic_1k.csv")

# 先读列头
stat_head = pd.read_csv(stat_path, nrows=5)
print(f"\n  列数: {stat_head.shape[1]}")
print(f"  列名: {list(stat_head.columns)}")
print(f"\n  前5行预览:\n{stat_head.to_string(index=False)}")

# 分块统计：行数 + 数值均值
print(f"\n  分块统计中（chunksize=300,000）...")
t0 = time.time()
total_stat_rows = 0
video_id_min, video_id_max = float('inf'), float('-inf')
counts_sum = 0

numeric_cols = ["show_cnt", "play_cnt", "valid_play_cnt", "like_cnt",
                "comment_cnt", "follow_cnt", "share_cnt", "play_progress"]
running_means = {c: [] for c in numeric_cols}

for chunk in pd.read_csv(stat_path, chunksize=300_000,
                          dtype={"video_id": "int64", "counts": "int64"}):
    n = len(chunk)
    total_stat_rows += n
    video_id_min = min(video_id_min, chunk["video_id"].min())
    video_id_max = max(video_id_max, chunk["video_id"].max())
    counts_sum += chunk["counts"].sum()
    for col in numeric_cols:
        if col in chunk.columns:
            running_means[col].append(chunk[col].mean())

elapsed = time.time() - t0
print(f"  完整行数: {total_stat_rows:,}  (耗时 {elapsed:.1f}s)")
print(f"  video_id 范围: {video_id_min} → {video_id_max}")
print(f"  总统计条目数 (counts之和): {int(counts_sum):,}")
print(f"\n  关键指标全局均值:")
for col in numeric_cols:
    vals = running_means[col]
    if vals:
        print(f"    {col:<28}: {np.mean(vals):.4f}")


# ──────────────────────────────────────────────
# 7. 数据集整体总结
# ──────────────────────────────────────────────
section("7. KuaiRand-1K 整体总结")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │               KuaiRand-1K 数据集概览                    │
  ├─────────────────────────────────────────────────────────┤
  │  用户数量            : 1,000 (user_features)            │
  │  视频数量            : ~4,369,953 (README)              │
  │  随机曝光日志行数    : ~43,028                          │
  │  标准日志 Phase1 行数: ~{total_rows:,}              │
  │  标准日志 Phase2 行数: ~{total_rows2:,}              │
  │  视频基础特征行数    : ~{total_videos:,}               │
  │  视频统计特征行数    : ~{total_stat_rows:,} (3.2GB)         │
  ├─────────────────────────────────────────────────────────┤
  │  用户特征维度        : 30 列 (含加密one-hot)            │
  │  视频基础特征列数    : 12 列 (含tag多标签)              │
  │  视频统计特征列数    : {stat_head.shape[1]} 列 (日均统计量)            │
  │  交互反馈信号数      : 12 种 (click/like/follow等)      │
  ├─────────────────────────────────────────────────────────┤
  │  时间跨度            : 2022-04-08 ~ 2022-05-08 (1个月) │
  │  随机干预比例        : ~0.37%                           │
  │  推荐场景数 (tab)    : 15 种策略                        │
  └─────────────────────────────────────────────────────────┘

  【对项目的影响分析】
  A组 (数据基座):
    - video_features_statistic 3.2GB 不可全量加载，需分块处理或
      预先聚合为每个video_id一行后存为 Parquet/HDF5
    - tag字段是多标签字符串，需解析为Multi-hot向量供B组使用
    - 时间戳 time_ms 精确到毫秒，直接支持C组的时间衰减计算

  B组 (冷启动):
    - user one-hot特征(30维)可直接用于人口统计KNN
    - video tag字段 → Multi-hot编码 → 余弦相似度冷启动
    - 随机日志(log_random)是无偏数据，适合做对比评估基准

  C组 (兴趣演化):
    - 标准日志 Phase1+2 共~1100万条，时序完整，天然支持Markov链
    - time_ms 字段支持精细化时间衰减建模
    - 每用户平均约11,713条交互，序列足够长
""")

print("分析完成！")
