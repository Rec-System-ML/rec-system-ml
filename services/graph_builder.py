from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Module 2：興趣演化圖核心算法
#
# 職責：給定單個用戶的完整互動歷史，輸出一張「興趣演化圖」——
#   • InterestNode：每個興趣話題（標籤）的節點，帶狀態（active/fading/mutation/predicted）
#   • InterestLink：興趣話題之間的跳轉邊，帶一階馬可夫轉移概率
#
# 整體流水線（10步）：
#   [01] 篩選目標用戶，按時間排序
#   [02] 展開視頻→標籤映射，構建長格式互動表
#   [03] 按標籤聚合基礎統計（互動次數、首/末時間、點擊、觀看時長）
#   [04] 計算指數時間衰減分 decay = exp(−λ × Δdays)
#   [05] 首現時刻正規化到 [0.05, 0.95]，用於 D3.js 橫軸定位
#   [06] 突變檢測：Z 檢驗（3天短窗口 vs 90天長窗口占比差異）
#   [07] 節點狀態分類：mutation > active(60%分位) > fading
#   [08] 預測節點：從全庫視頻中找與 active 標籤共現最多的、用戶未接觸的新話題
#   [09] 序列轉移概率：相鄰視頻標籤對計數 → 一階馬可夫轉移概率 → InterestLink
#   [10] 組裝 InterestNode（含 ENG/CTR/WATCH 指標）→ 返回給 D3.js 渲染器
# ═══════════════════════════════════════════════════════════════════════════════

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_shared = (Path(__file__).resolve().parent.parent / "../recsys-shared").resolve()
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from interest_graph.graph_data import InterestLink, InterestNode
from tag_display import get_tag_display_name


def build_interest_graph(
    user_id: int,
    interactions_df: pd.DataFrame,
    tag_matrix: pd.DataFrame,
    lambda_decay: float = 0.035,
    mutation_z_threshold: float = 2.0,
    top_k_tags: int = 10,
    top_k_edges: int = 12,
) -> tuple[list[InterestNode], list[InterestLink]]:
    """Build interest-evolution graph data for a single user.

    Parameters
    ----------
    user_id : int
        Target user whose interest graph will be constructed.
    interactions_df : pd.DataFrame
        Full interaction log containing at least ``user_id``, ``video_id``,
        and a time column (``time_ms`` preferred).
    tag_matrix : pd.DataFrame
        Multi-hot matrix indexed by ``video_id``; columns are integer tag IDs,
        values are 0/1.
    lambda_decay : float
        指數衰減率，單位：1/天。默認 0.035，對應半衰期 ≈ 20 天。
        計算：half_life = ln(2) / 0.035 ≈ 20 天——即某話題 20 天沒互動，
        其「新鮮度」降至原來的 50%。
    mutation_z_threshold : float
        Z 分數閾值，超過此值判定為興趣突變（mutation）。默認 2.0，
        對應二項比例檢驗約 95% 置信水平，避免過多誤報。
    top_k_tags : int
        返回的最大節點數。默認 10，平衡圖可讀性與覆蓋度。
    top_k_edges : int
        返回的最大邊數（轉移概率最高的前 K 條）。默認 12。
    """

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 01] 篩選目標用戶，按時間排序
    #
    # 目的：從全量日誌（最多 25 萬行）中取出本用戶的所有互動記錄，
    # 並按時間戳從早到晚排列——後續的「序列」分析依賴嚴格的時間順序。
    # ══════════════════════════════════════════════════════════════════════════
    udf = interactions_df[interactions_df["user_id"] == user_id].copy()
    if udf.empty:
        return [], []

    # 自動探測時間列名（優先用毫秒時間戳 time_ms，也兼容其他列名）
    time_col = next(
        (c for c in ("time_ms", "timestamp", "ts", "date") if c in udf.columns),
        None,
    )
    if time_col is None:
        # 無時間列時退化為行順序，保證後續邏輯不崩潰
        udf["_seq"] = range(len(udf))
        time_col = "_seq"
    udf = udf.sort_values(time_col).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 02] 視頻 → 標籤映射：構建長格式互動表
    #
    # tag_matrix 是一個多熱矩陣（video × tag，值為 0/1）。
    # 這裡將其「展開」（melt）為長格式：每行 = (video_id, tag_id)，
    # 再與用戶互動記錄合併，得到：(user_id, video_id, tag_id, time_ms, ...)。
    #
    # 舉例：視頻 #12345 帶標籤 [美食, 旅行]，用戶點擊了它，
    # 展開後產生兩行記錄，分別對應「美食」和「旅行」的互動事件。
    # ══════════════════════════════════════════════════════════════════════════
    valid_vids = tag_matrix.index.intersection(udf["video_id"].unique())
    if valid_vids.empty:
        return [], []

    sub = tag_matrix.loc[valid_vids].copy()
    sub.index.name = "video_id"

    # melt：寬表 → 長表；query("_v == 1")：只保留該視頻確實帶有的標籤
    tag_long = (
        sub.reset_index()
        .melt(id_vars="video_id", var_name="tag_id", value_name="_v")
        .query("_v == 1")
        .drop(columns="_v")
    )
    tag_df = udf.merge(tag_long, on="video_id", how="inner")
    if tag_df.empty:
        return [], []

    tag_df["tag_id"] = tag_df["tag_id"].astype(int)

    # 記錄全局時間跨度，後續正規化用
    min_t = float(tag_df[time_col].min())   # 最早互動時間（毫秒）
    max_t = float(tag_df[time_col].max())   # 最晚互動時間（毫秒）
    t_range = max_t - min_t if max_t > min_t else 1.0

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 03] 按標籤聚合基礎統計
    #
    # 對每個標籤（tag_id）計算：
    #   count      - 總互動次數（衡量歷史興趣深度）
    #   last_time  - 最後一次互動時間（衡量興趣新鮮度，用於 Step 04/07）
    #   first_time - 第一次互動時間（用於 Step 05 定位 X 軸）
    #   is_click / is_like / ... - 各類互動信號的累計次數
    #   avg_play_ms - 平均觀看時長（毫秒），用於計算 WATCH 指標
    # ══════════════════════════════════════════════════════════════════════════
    aggs: dict[str, tuple[str, str]] = {
        "count":     (time_col, "size"),
        "last_time": (time_col, "max"),
        "first_time":(time_col, "min"),
    }
    eng_cols_present: list[str] = []
    for c in ("is_click", "is_like", "is_follow", "is_comment", "is_forward"):
        if c in tag_df.columns:
            aggs[c] = (c, "sum")
            eng_cols_present.append(c)
    if "play_time_ms" in tag_df.columns:
        aggs["avg_play_ms"] = ("play_time_ms", "mean")

    tag_stats = tag_df.groupby("tag_id").agg(**aggs).reset_index()

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 04] 指數時間衰減：decay = exp(−λ × Δdays)
    #
    # 公式含義：
    #   Δdays = (max_t - last_time) / 86_400_000
    #           → 86_400_000 = 24h × 60min × 60s × 1000ms，毫秒轉天數
    #   decay ∈ [0, 1]：值越接近 1 表示興趣越新鮮，越接近 0 表示越陳舊
    #
    # 示例（λ=0.035）：
    #   昨天看的 Δdays=1  → decay = e^(-0.035×1)  ≈ 0.966（非常新鮮）
    #   上月看的 Δdays=30 → decay = e^(-0.035×30) ≈ 0.350（有些過時）
    #   兩月前  Δdays=60 → decay = e^(-0.035×60) ≈ 0.122（幾乎淡出）
    #
    # 用途：decay 值傳給 D3.js 控制節點大小——越新鮮的節點視覺上越大。
    # ══════════════════════════════════════════════════════════════════════════
    tag_stats["delta_days"] = (max_t - tag_stats["last_time"]) / 86_400_000.0
    tag_stats["decay"] = np.exp(-lambda_decay * tag_stats["delta_days"]).clip(0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 05] 首現時刻正規化 → [0.05, 0.95]
    #
    # 將每個標籤「第一次出現的時間」線性壓縮到 [0.05, 0.95]，
    # 作為 D3.js 力導向圖的橫軸座標（timestamp 屬性）：
    #   - 靠左（≈0.05）：很早就培養的興趣（老興趣）
    #   - 靠右（≈0.95）：最近才出現的新興趣
    #
    # 首尾各留 5% 邊距，防止節點頂到畫布邊緣。
    # ══════════════════════════════════════════════════════════════════════════
    tag_stats["timestamp"] = 0.05 + 0.9 * (tag_stats["first_time"] - min_t) / t_range

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 06] 突變檢測：二項比例 Z 檢驗
    #
    # 「突變」定義：某標籤在最近 3 天的互動占比，相對其 90 天歷史基線
    # 出現統計顯著的暴增。
    #
    # 公式：
    #   r_short = 該標籤近 3 天互動次數 / 近 3 天全部標籤互動總數
    #   r_long  = 該標籤近 90 天互動次數 / 近 90 天全部標籤互動總數
    #   σ_long  = sqrt(r_long × (1 - r_long) / n_90d)  ← 二項分佈標準誤
    #   z = (r_short - r_long) / σ_long
    #
    # 判定條件：z > 2.0（約 95% 置信）且 90 天歷史計數 < 5（新興話題）
    # 加「歷史計數 < 5」的保護：避免歷史記錄多的標籤因 σ 極小而虛高 z 值。
    # ══════════════════════════════════════════════════════════════════════════
    short_cut = max_t - 3  * 86_400_000   # 3 天前的時間戳（毫秒）
    long_cut  = max_t - 90 * 86_400_000   # 90 天前的時間戳（毫秒）

    # 分別統計每個標籤在短/長窗口內的互動次數
    s_cnt = tag_df.loc[tag_df[time_col] >= short_cut].groupby("tag_id").size()
    l_cnt = tag_df.loc[tag_df[time_col] >= long_cut ].groupby("tag_id").size()
    tot_s = max(1, int(s_cnt.sum()))   # 短窗口總互動數（防止除零）
    tot_l = max(1, int(l_cnt.sum()))   # 長窗口總互動數

    mutation_ids: set[int] = set()
    for tid in s_cnt.index:
        short_rate = s_cnt.get(tid, 0) / tot_s          # 短窗口占比
        long_count = l_cnt.get(tid, 0)
        long_rate  = long_count / tot_l                  # 長窗口占比（歷史基線）

        # 二項標準誤：max(0.01, ...) 防止分母為 0
        denom = max(0.01, math.sqrt(long_rate * (1 - long_rate) / max(1, long_count)))
        z = (short_rate - long_rate) / denom

        # 雙重條件：統計顯著 AND 歷史記錄少（確為新爆發的興趣）
        if z > mutation_z_threshold and long_count < 5:
            mutation_ids.add(int(tid))

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 07] 節點狀態分類：mutation > active > fading
    #
    # 分類邏輯（優先級從高到低）：
    #   1. mutation（紅色）：Step 06 中 z > 2.0 的標籤，優先級最高，覆蓋其他狀態
    #   2. active（綠色）：last_time 排名在前 60% 分位以上（即比較近期互動的標籤）
    #   3. fading（灰色）：其餘，即最後互動時間相對較舊的標籤
    #
    # 為何用「60% 分位排名」而非固定 decay 閾值？
    #   稀疏用戶的所有記錄可能集中在一個月前，所有 decay 都偏低；
    #   用相對排名能保證：不論用戶活躍或稀疏，圖上永遠有約 40% 節點是綠色，
    #   視覺信息量穩定，不出現「全灰圖」的退化情況。
    #
    # 示例（5個標籤按 last_time 排名）：
    #   搞笑  4月22日 → 20% → fading
    #   遊戲  4月24日 → 40% → fading
    #   音樂  4月28日 → 60% → active  ← 60% 分位線
    #   美食  5月1日  → 80% → active
    #   旅行  5月6日  →100% → active
    # ══════════════════════════════════════════════════════════════════════════
    tag_stats["status"] = "fading"   # 默認全部 fading
    tag_stats["_recency_pct"] = tag_stats["last_time"].rank(method="average", pct=True)
    tag_stats.loc[tag_stats["_recency_pct"] >= 0.60, "status"] = "active"
    tag_stats.loc[tag_stats["tag_id"].isin(mutation_ids), "status"] = "mutation"

    # Guardrail：當所有非 mutation 標籤時間戳幾乎相同時（如只有 1 天數據），
    # 60% 分位會讓所有節點都是 active，圖失去對比度。
    # 強制保留最舊的 30% 非 mutation 標籤為 fading。
    non_mut_mask = ~tag_stats["tag_id"].isin(mutation_ids)
    if non_mut_mask.sum() >= 3 and tag_stats.loc[non_mut_mask, "status"].eq("active").all():
        oldest_n = max(1, int(non_mut_mask.sum() * 0.3))
        oldest_idx = tag_stats.loc[non_mut_mask].nsmallest(oldest_n, "last_time").index
        tag_stats.loc[oldest_idx, "status"] = "fading"

    # 取互動次數最多的前 top_k_tags 個標籤作為主節點集合
    top = tag_stats.nlargest(top_k_tags, "count").copy()

    # 保證 top 中至少有一個 fading 節點（灰色），讓圖更有對比度；
    # 若 top 全為 active/mutation，從全量標籤中找最熱的 fading 節點替換最弱的 active
    if not top["status"].eq("fading").any():
        fading_pool = tag_stats.loc[tag_stats["status"] == "fading"].sort_values(
            "count", ascending=False
        )
        if not fading_pool.empty:
            replace_idx = top.loc[top["status"] != "mutation", "count"].idxmin()
            top.loc[replace_idx, top.columns] = fading_pool.iloc[0][top.columns].values

    top_set: set[int] = set(top["tag_id"])
    active_set: set[int] = set(top.loc[top["status"] == "active", "tag_id"])
    user_all_tags: set[int] = set(tag_stats["tag_id"])   # 用戶歷史接觸過的所有標籤

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 08] 預測節點（紫色）：鄰居標籤共現擴充
    #
    # 算法：「標籤共現即相關」—— Item-Based Collaborative Filtering 的簡化版
    #
    # 三步邏輯：
    #   1. 找出用戶當前所有 active 標籤集合（如 {美食, 旅行}）
    #   2. 在全量視頻庫（tag_matrix）中，找到包含任一 active 標籤的視頻，
    #      統計這些視頻還帶有哪些其他標籤及其出現次數（共現頻率）
    #   3. 優先選擇「用戶從未接觸過的」高共現標籤，標記為 predicted
    #      → 代表算法預測的下一個潛在興趣話題
    #
    # 降級策略（當用戶已是重度用戶，覆蓋了大部分標籤）：
    #   優先：用戶從未看過的標籤（user_all_tags 之外）
    #   其次：用戶看過但當前不在 active 的標籤（active_set 之外）
    #   兜底：全庫最熱門的標籤（任何情況下都能填充 predicted 節點）
    #
    # 預測節點的固定屬性：
    #   decay = 0.65（中等新鮮度，表示「尚未發生但可能性高」）
    #   timestamp = 0.85（靠右，表示面向未來）
    #   count = 0（用戶尚未有任何此標籤的互動記錄）
    # ══════════════════════════════════════════════════════════════════════════
    if active_set:
        cols_active = [c for c in active_set if c in tag_matrix.columns]
        if cols_active:
            # 找包含至少一個 active 標籤的視頻（布爾掩碼）
            active_mask = tag_matrix[cols_active].max(axis=1) == 1
            # 統計這些視頻的所有標籤出現次數（共現頻率）
            neighbor_sums = tag_matrix.loc[active_mask].sum(axis=0)
            neighbor_sums = neighbor_sums[neighbor_sums > 0].sort_values(ascending=False)

            # 優先推用戶從未接觸過的標籤
            predicted_candidates = [
                int(c) for c in neighbor_sums.index if int(c) not in user_all_tags
            ]
            # 降級：用戶看過但不在 active 的標籤
            if not predicted_candidates:
                predicted_candidates = [
                    int(c) for c in neighbor_sums.index if int(c) not in active_set
                ]
            # 兜底：全庫熱門標籤
            if not predicted_candidates:
                global_pop = tag_matrix.sum(axis=0).sort_values(ascending=False)
                predicted_candidates = [
                    int(c) for c in global_pop.index if int(c) not in active_set
                ]

            # 填充剩餘 slot（至少 2 個預測節點）
            slots = max(2, top_k_tags - len(top))
            for ptid in predicted_candidates[:slots]:
                pred_row: dict = {
                    "tag_id":     ptid,
                    "count":      0,
                    "last_time":  max_t,
                    "first_time": max_t,
                    "delta_days": 0.0,
                    "decay":      0.65,   # 固定中等新鮮度
                    "timestamp":  0.85,   # 固定靠右（面向未來）
                    "status":     "predicted",
                }
                for c in eng_cols_present:
                    pred_row[c] = 0
                if "avg_play_ms" in top.columns:
                    pred_row["avg_play_ms"] = 0.0
                top = pd.concat([top, pd.DataFrame([pred_row])], ignore_index=True)
            top_set = set(top["tag_id"])

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 09] 序列轉移概率 → InterestLink（一階馬可夫鏈）
    #
    # 「一階馬可夫」：只用「上一個視頻的標籤」預測「當前視頻的興趣標籤」，
    # 不考慮更早的歷史（無記憶性假設）。
    #
    # 計算過程：
    #   1. 構建 vid_to_tags 字典：video_id → 該視頻在 top_set 中的標籤列表
    #   2. 按時間順序遍歷用戶的觀看序列，記錄相鄰視頻間的標籤跳轉次數
    #      trans[s][t] += 1 表示「看完帶標籤 s 的視頻後，下一個視頻帶標籤 t」
    #   3. 轉移概率：P(s→t) = count(s→t) / Σ_t' count(s→t')
    #
    # 示例：觀看序列 [美食] → [旅行] → [美食] → [旅行] → [音樂]
    #   trans[美食][旅行] = 2, trans[旅行][美食] = 1, trans[美食][音樂] = 1（最後一跳）
    #   等待...實際序列：美食→旅行, 旅行→美食, 美食→旅行（不含最後），美食→音樂？
    #   P(美食→旅行) = 2/3 ≈ 0.67, P(美食→音樂) = 1/3 ≈ 0.33
    #
    # 最後只保留概率最高的 top_k_edges 條邊（默認 12），
    # 即圖上最粗的有向連接線，代表最頻繁的興趣遷移路徑。
    # ══════════════════════════════════════════════════════════════════════════

    # 建立 video → top_set 標籤 的映射字典（只關心在展示集合中的標籤）
    vid_to_tags: dict[int, list[int]] = {}
    for vid in udf["video_id"].unique():
        if vid not in tag_matrix.index:
            continue
        row = tag_matrix.loc[vid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        tags = [int(t) for t in row[row == 1].index if int(t) in top_set]
        if tags:
            vid_to_tags[int(vid)] = tags

    # 按時間順序掃描用戶觀看序列，統計相鄰標籤跳轉次數
    trans: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    prev_tags: list[int] | None = None
    for vid in udf["video_id"]:
        cur = vid_to_tags.get(int(vid))
        if cur is not None and prev_tags is not None:
            for s in prev_tags:   # 上一個視頻的標籤
                for t in cur:     # 當前視頻的標籤
                    if s != t:
                        trans[s][t] += 1   # 記錄跳轉次數
        if cur is not None:
            prev_tags = cur       # 更新「上一個視頻的標籤」

    # 計算轉移概率，構建 InterestLink 列表
    # timestamp 取源節點和目標節點首現時刻的平均值（用於 D3.js 定位）
    ts_map = dict(zip(top["tag_id"].astype(int), top["timestamp"].astype(float)))
    links: list[InterestLink] = []
    for src, tgts in trans.items():
        total = sum(tgts.values())          # 從 src 出發的總跳轉次數
        for tgt, cnt in tgts.items():
            prob = cnt / total              # P(src→tgt)
            avg_ts = (ts_map.get(src, 0.5) + ts_map.get(tgt, 0.5)) / 2.0
            links.append(InterestLink(str(src), str(tgt), prob, timestamp=avg_ts))

    # 按概率降序排列，只保留最強的 top_k_edges 條邊
    links.sort(key=lambda lk: lk.probability, reverse=True)
    links = links[:top_k_edges]

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 10] 組裝 InterestNode 列表，附加 active 節點的詳細指標
    #
    # 每個 InterestNode 包含：
    #   id        - 標籤 ID（字符串）
    #   label     - 標籤顯示名稱（如「美食」，由 get_tag_display_name 映射）
    #   decay     - Step 04 計算的新鮮度分（控制 D3.js 節點大小）
    #   status    - Step 07/08 分類的狀態（決定節點顏色）
    #   timestamp - Step 05 正規化的首現時刻（決定節點橫軸位置）
    #   metrics   - 僅 active 節點有：ENG / CTR / WATCH 三個指標
    #
    # 節點顏色映射（在 D3.js 渲染器中定義）：
    #   🟢 active    → 綠色
    #   🔴 mutation  → 紅色
    #   🟣 predicted → 紫色
    #   ⚫ fading    → 灰色
    #
    # 指標計算（僅 active 節點）：
    #   ENG（綜合互動率）= (like+follow+comment+forward 總次數) / count × 100%
    #   CTR（點擊率）    = is_click 總次數 / count × 100%
    #   WATCH（平均觀看）= avg_play_ms / 1000（毫秒轉秒）
    # ══════════════════════════════════════════════════════════════════════════
    _eng_metric_cols = [
        c for c in ("is_like", "is_follow", "is_comment", "is_forward") if c in top.columns
    ]

    nodes: list[InterestNode] = []
    for _, r in top.iterrows():
        tid = int(r["tag_id"])
        metrics = None
        if r["status"] == "active" and r["count"] > 0:
            eng   = sum(float(r[c]) for c in _eng_metric_cols) / r["count"] * 100
            ctr   = float(r.get("is_click", 0)) / r["count"] * 100
            watch = float(r.get("avg_play_ms", 0)) / 1000.0
            metrics = {
                "ENG":   f"{eng:.0f}%",
                "CTR":   f"{ctr:.0f}%",
                "WATCH": f"{watch:.0f}s",
            }
        nodes.append(
            InterestNode(
                id=str(tid),
                label=get_tag_display_name(tid),
                decay=float(r["decay"]),
                status=r["status"],
                tags=[tid],
                timestamp=float(r["timestamp"]),
                metrics=metrics,
            )
        )

    return nodes, links


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷加載器：從 KuaiRand-1K 數據目錄直接構建興趣圖
# ═══════════════════════════════════════════════════════════════════════════════


def build_demo_from_kuairand(
    data_dir: str | Path,
    user_id: int | None = None,
    sample_rows: int = 250_000,
) -> tuple[list[InterestNode], list[InterestLink]]:
    """Load KuaiRand-1K data and build an interest graph.

    如果 user_id 為 None，自動選取互動次數最多的用戶。
    sample_rows 默認 250,000——原始 CSV 約 665 萬行，截取前 25 萬行以控制計算成本
    （可通過 python main.py --rows 6000000 加載更多）。
    """
    from data_pipeline import load_kuairand_tables, parse_tag_ids
    from sklearn.preprocessing import MultiLabelBinarizer
    from tag_display import ensure_tag_mapping

    tables = load_kuairand_tables(data_dir)
    ensure_tag_mapping(data_dir)
    interactions = tables.interactions.copy()

    if sample_rows and sample_rows < len(interactions):
        interactions = interactions.head(sample_rows)

    if user_id is None:
        # 自動選最活躍用戶（互動次數最多）
        user_id = int(interactions["user_id"].value_counts().idxmax())

    items = tables.items.copy()
    tag_col = next(
        (c for c in ("tag", "tags", "tag_ids", "video_tag") if c in items.columns),
        None,
    )
    if tag_col is None:
        raise KeyError("No tag column found in items table")

    # 將每個視頻的標籤列表解析並用 MultiLabelBinarizer 轉為多熱矩陣
    # 多熱矩陣形狀：(視頻數量, 標籤詞彙表大小)，值為 0/1
    items["_tags"] = items[tag_col].apply(parse_tag_ids)
    items = items.drop_duplicates(subset=["video_id"])

    mlb = MultiLabelBinarizer()
    vectors = mlb.fit_transform(items["_tags"])
    tag_matrix = pd.DataFrame(vectors, index=items["video_id"], columns=mlb.classes_)
    tag_matrix.index.name = "video_id"

    return build_interest_graph(
        user_id=user_id,
        interactions_df=interactions,
        tag_matrix=tag_matrix,
    )
