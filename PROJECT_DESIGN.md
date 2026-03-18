# KuaiRand-1K 推荐系统 · 技术实现文档

**版本**: v2.0（已实现）| **日期**: 2026-03-18 | **数据集**: KuaiRand-1K

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [数据集与预处理](#2-数据集与预处理)
3. [后端实现](#3-后端实现)
4. [推荐算法](#4-推荐算法)
5. [兴趣演化模块](#5-兴趣演化模块)
6. [前端实现](#6-前端实现)
7. [API 接口文档](#7-api-接口文档)

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────┐
│               前端 (纯 HTML/CSS/JS)              │
│  index.html           interest_evolution.html   │
│  · Dashboard          · D3.js 力导向图           │
│  · User Profile       · 粒子流动效果             │
│  · Live Simulate      · 时间轴 + 突变检测         │
└────────────────────┬────────────────────────────┘
                     │ REST / WebSocket
┌────────────────────▼────────────────────────────┐
│              后端 (FastAPI + Uvicorn)            │
│  /api/users          /api/popular               │
│  /api/user/{id}/profile                         │
│  /api/recommend      /api/recommend/realtime    │
│  /api/cold_start     /api/stats                 │
│  ws://host/ws/recommend                         │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              模型 Artifact (.joblib)             │
│  ItemKNN  ·  XGBoost CTR  ·  TimeDecayReranker  │
└─────────────────────────────────────────────────┘
```

---

## 2. 数据集与预处理

### 2.1 KuaiRand-1K 数据集

| 文件 | 说明 | 关键统计 |
|------|------|---------|
| `log_standard_4_08_to_4_21_1k.csv` | Phase1 训练日志 | CTR 42.6%，长观看率 30.5% |
| `log_standard_4_22_to_5_08_1k.csv` | Phase2 训练日志 | CTR 31.2%，长观看率 22.1% |
| `log_random_4_22_to_5_08_1k.csv` | 随机曝光（无偏测试集） | CTR 17.4%，43,028 条 |
| `user_features_1k.csv` | 1,000 用户特征 | 31 列，含活跃度/社交/加密特征 |
| `video_features_basic_1k.csv` | 视频基础特征 | tag 平均 1.23 个/视频 |

Phase1→Phase2 CTR 下降 **11.4pp**，验证了时间衰减重排的必要性。随机曝光日志作为无偏基线（is_rand=1）。

### 2.2 训练流程

运行 `back/api.py` 完成以下步骤：

```
1. 加载 Phase1 + Phase2 标准日志 → 构建训练交互矩阵
2. 训练 ItemKNN（物品协同过滤）
3. 构建特征（user_id, video_id, time_ms, knn_score）
4. 训练 XGBoost CTR 预测模型
5. 在随机曝光日志上评估（HR@10, NDCG@10, Precision@10, Recall@10）
6. 将所有模型序列化 → back/checkpoints/mvp_artifact.joblib
```

---

## 3. 后端实现

### 3.1 目录结构

```
back/
├── main.py                  # FastAPI 入口，注册路由
├── api.py                   # 训练脚本，生成 artifact
├── core/
│   └── loader.py            # artifact 加载、特征工程、工具函数
├── models/
│   ├── item_knn.py          # ItemKNN 物品协同过滤
│   └── xgboost_ctr.py       # XGBoost CTR 预测
├── routers/
│   ├── user.py              # /api/users, /api/user/{id}/profile
│   ├── recommend.py         # /api/recommend, /api/recommend/realtime, WebSocket
│   ├── cold_start.py        # /api/cold_start
│   └── stats.py             # /api/stats
└── shared/
    ├── data_pipeline.py     # 数据读取与清洗
    ├── evaluation.py        # HR@K / NDCG@K 评估
    ├── reranker.py          # 时间衰减重排器
    └── tag_display.py       # tag_id → 名称映射
```

### 3.2 Artifact 结构

`mvp_artifact.joblib` 中包含：

| Key | 类型 | 说明 |
|-----|------|------|
| `item_knn` | ItemKNN | 物品协同过滤模型，含 `user_histories`、`item_neighbor_scores` |
| `ctr_model` | XGBoostCTR | CTR 预测模型 |
| `feature_builder` | FeatureBuilder | 特征工程转换器 |
| `reranker` | TimeDecayReranker | 时间衰减重排器（λ=0.1/day） |
| `train_users` | list[int] | 训练集用户 ID 列表 |

---

## 4. 推荐算法

### 4.1 ItemKNN 物品协同过滤

```
候选生成：
  · 用户历史点击视频 → 查找每个视频的 top-K 相似视频
  · 补充全站热门 top-800 兜底

打分：
  knn_score = Σ(sim(hist_vid, candidate) for hist_vid in user_history)
```

### 4.2 XGBoost CTR 预测

特征包括：`user_id`, `video_id`, `time_ms`, `knn_score` 及统计特征。
在无偏随机测试集上评估，消除曝光偏差。

### 4.3 Blend 融合与时间衰减重排

```
prediction = α × ctr_score + (1-α) × sigmoid(knn_score)

时间衰减重排:
  rerank_score = prediction × exp(-λ × Δt_days)
  λ = 0.1，Δt 为视频距当前的时间差
```

默认 `α = 0.6`（XGBoost 权重），可通过 API 参数调整。

### 4.4 冷启动（新用户）

```
α 随点击数从 0 增长到 1：
  α_cold = min(n_clicks / max_clicks, 1.0)

score = (1-α) × pop_score + α × knn_score

· α=0：纯热门兜底（Pop Top10）
· α→1：纯 KNN 个性化
· 前端橙色/紫色色条实时反映两项贡献比例
```

### 4.5 老用户实时重排

Session 内新增点击叠加到历史，extra_weight 随点击数线性增长（最大 0.5），合并进 knn_score 重新排序。

---

## 5. 兴趣演化模块

### 5.1 数据生成

`gen_interest_data.py` 从 KuaiRand-1K 原始日志生成 `ui_prototype/interest_evolution_data.js`：

```
选取 3 个代表用户（full_active / high_active / mutation_active）
↓
按时间窗口划分为 6 个快照（每窗口约 5 天）
↓
计算每个快照内：
  · Tag 激活度（long_view 行为加权）
  · Markov 转移概率（相邻点击视频间的 tag 转移）
  · 时间衰减权重 w(t) = exp(-0.10 × Δt_days)
  · 突变检测：某 Tag 单窗口 share > 0.10 且之前从未出现 → mutation=true
```

### 5.2 突变检测逻辑

```python
# 新 Tag 首次出现且份额超过阈值
new_tags = current_tags - prev_tags
for tag in new_tags:
    if tag_share[tag] > MUTATION_THRESHOLD:  # 0.10
        mark_mutation(tag)
```

用户 #546 在 t=1（2022-04-14）窗口有确认的突变事件。

### 5.3 前端可视化层次

```
z-index 0: #particles canvas    — 背景星点动画（70 颗漂浮粒子）
z-index 1: #mainSvg             — D3.js 力导向图（节点 + 边）
z-index 2: #flow-canvas canvas  — 粒子流动（沿 Markov 转移边）
z-index 3: #ui div              — 面板、时间轴、控制按钮
```

### 5.4 粒子流动规则

- 每条可见边生成 1~6 颗粒子，数量与转移概率正比
- 粒子速度 = 0.0025 + prob × 0.005（高概率边流得更快）
- 粒子带渐隐拖影，视觉上呈"流光"效果
- 若连线任意一端为突变节点，粒子颜色变为红色（`#ff3355`）
- 快照切换时，消失边的粒子自动移除，新出现边补充粒子

---

## 6. 前端实现

### 6.1 技术栈

| 组件 | 技术 |
|------|------|
| UI 框架 | 纯 HTML5 + CSS3 + Vanilla JS（无框架） |
| 力导向图 | D3.js v7 |
| 统计图表 | Chart.js |
| 粒子动画 | Canvas 2D API |
| 图标 | Bootstrap Icons (CDN) |
| 字体 | Courier New（科幻等宽风格） |

### 6.2 主题配色

```css
--bg:      #020b18   /* 深海蓝背景 */
--accent:  #00e5a0   /* 绿色主题色 */
--accent2: #5B4FE8   /* 紫色（KNN/个性化） */
--mut:     #ff3355   /* 红色（突变事件） */
--pred:    #8866ff   /* 预测节点 */
```

### 6.3 页面模块

**`index.html` — 主 Dashboard**

| 页面 | 功能 |
|------|------|
| Overview | KPI 卡、CTR 信号对比柱状图、模型 Metrics |
| User Profile | 用户活跃度、历史兴趣 Tag 环形图、KNN 散点图 |
| Live Simulate | 新用户冷启动（Pop→KNN 色条渐变）+ 老用户实时重排 |

**`interest_evolution.html` — 兴趣演化可视化**

- D3.js 力导向图，节点按激活度/突变/预测三态渲染
- 时间轴滑块，支持自动播放
- 突变位置显示红色菱形标记（◆ MUTATION）
- EVO PROB 指标实时更新
- 全屏可嵌入 Dashboard iframe 预览

### 6.4 API 降级策略

所有 `apiFetch` 调用在后端离线时静默降级到静态数据，前端可独立演示。

---

## 7. API 接口文档

### 用户相关

| 方法 | 路径 | 返回 |
|------|------|------|
| GET | `/api/users` | 训练集用户 ID 列表（max 200） |
| GET | `/api/user/{id}/profile` | 点击数、活跃度百分位、Tag 分布 |

### 推荐相关

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/popular?n=10` | 全站热门 Top-N（固定，不依赖用户） |
| POST | `/api/recommend` | ItemKNN + XGBoost Blend 推荐 |
| POST | `/api/recommend/realtime` | 老用户实时重排（叠加 session 点击） |
| POST | `/api/cold_start` | 新用户冷启动（α 随点击数增长） |
| GET | `/api/stats` | 模型评估指标（HR@10, NDCG@10 等） |
| WS | `/ws/recommend` | WebSocket 实时流式推荐 |

### 请求示例

```json
// POST /api/recommend
{
  "user_id": 546,
  "top_k": 10,
  "alpha": 0.6
}

// POST /api/cold_start
{
  "clicked_ids": [1234, 5678],
  "top_k": 10
}

// POST /api/recommend/realtime
{
  "user_id": 970,
  "extra_clicks": [1234],
  "top_k": 10,
  "alpha": 0.6
}
```

### 响应示例

```json
// GET /api/user/546/profile
{
  "user_id": 546,
  "click_count": 1534,
  "activity_percentile": 78,
  "tag_profile": [
    {"tag_id": 12, "name": "Gaming", "pct": 34.2}
  ],
  "history_sample": [10001, 10023, ...]
}

// POST /api/cold_start
{
  "results": [
    {
      "rank": 1,
      "video_id": 10001,
      "score": 0.8234,
      "pop_score": 0.76,
      "knn_score": 0.91,
      "alpha": 0.42,
      "tags": ["Gaming", "Tech"]
    }
  ]
}
```

---

## 附录：启动命令

```bash
# 1. 安装依赖
cd back && pip install -r requirements.txt

# 2. 训练模型（约 30 分钟，已有 checkpoints 可跳过）
python api.py

# 3. 启动后端
uvicorn main:app --reload --port 8000

# 4. 生成兴趣演化数据（可选，已有预生成文件）
cd .. && python gen_interest_data.py

# 5. 打开前端
cd ui_prototype && python -m http.server 5500
# 访问 http://localhost:5500
```
