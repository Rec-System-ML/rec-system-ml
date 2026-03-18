# KuaiRand-1K 推荐系统可视化平台

基于 KuaiRand-1K 数据集构建的冷启动 + 兴趣演化推荐系统，含完整后端 API 与交互式前端 Dashboard。

---

## 功能模块

| 模块 | 说明 |
|------|------|
| **Overview Dashboard** | KPI 统计、CTR 信号对比、模型综合指标 |
| **User Profile** | 用户活跃度、历史兴趣 Tag 分布（环形图）、KNN 邻域散点 |
| **Live Simulate** | 新用户冷启动（Pop → KNN 渐进个性化）+ 老用户实时重排 |
| **Interest Evolution** | D3.js 力导向图 · Markov 转移概率 · 时间衰减 · 粒子流动效果 · 突变检测 |

---

## 技术栈

**后端**
- FastAPI + Uvicorn
- ItemKNN（物品协同过滤）+ XGBoost CTR 预测
- 时间衰减重排（`w(t) = exp(-λΔt)`）

**前端**
- 纯 HTML/CSS/JS（无框架）
- D3.js v7 力导向图
- Chart.js 统计图表
- Canvas 粒子动画

---

## 数据集

使用 [KuaiRand-1K](https://kuairand.com/) 数据集，需自行下载并放置到 `KuaiRand-1K/data/`：

```
KuaiRand-1K/data/
├── log_standard_4_08_to_4_21_1k.csv   # Phase1 训练日志
├── log_standard_4_22_to_5_08_1k.csv   # Phase2 训练日志
├── log_random_4_22_to_5_08_1k.csv     # 随机曝光（去偏评估）
├── user_features_1k.csv
├── video_features_basic_1k.csv
└── video_features_statistic_1k.csv
```

---

## 快速开始

### 1. 安装依赖

```bash
cd back
pip install -r requirements.txt
```

### 2. 训练模型 & 生成 Artifact

```bash
cd back
python main.py --data-dir ../KuaiRand-1K/data
```

> 训练 ItemKNN + XGBoost 并将 artifact 保存到 `back/checkpoints/mvp_artifact.joblib`（约 55 MB）。**已随仓库附带预训练版本，可跳过此步骤直接启动后端。**

### 3. 启动后端

```bash
cd back
python api.py
```

API 文档：[http://localhost:8100/docs](http://localhost:8100/docs)

### 4. 生成兴趣演化数据

```bash
python gen_interest_data.py
```

> 输出 `ui_prototype/interest_evolution_data.js`（已含预生成版本，可跳过）。

### 5. 打开前端

访问 [http://localhost:8100](http://localhost:8100)

---

## 项目结构

```
.
├── back/
│   ├── api.py                   # FastAPI 入口
│   ├── main.py                  # 训练 & 构建 artifact
│   ├── core/loader.py           # 数据加载与特征工程
│   ├── models/
│   │   ├── item_knn.py          # ItemKNN 协同过滤
│   │   └── xgboost_ctr.py       # XGBoost CTR 模型
│   ├── routers/
│   │   ├── user.py              # 用户画像 API
│   │   ├── recommend.py         # 推荐 & 实时重排 API
│   │   ├── cold_start.py        # 冷启动 API
│   │   └── stats.py             # 统计 API
│   └── shared/                  # 数据管道、评估、重排器
├── ui_prototype/
│   ├── index.html               # 主 Dashboard
│   ├── interest_evolution.html  # 兴趣演化可视化
│   └── interest_evolution_data.js  # 预生成用户演化数据
├── gen_interest_data.py         # 从数据集生成演化数据
├── KuaiRand-1K/                 # 数据集目录（data/ 需自行下载）
└── PROJECT_DESIGN.md            # 详细设计文档
```

---

## 主要 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/users` | 可用用户列表 |
| GET | `/api/user/{id}/profile` | 用户画像（Tag 分布、活跃度） |
| GET | `/api/popular` | 全站热门 Top-N |
| POST | `/api/recommend` | ItemKNN + XGBoost 推荐 |
| POST | `/api/recommend/realtime` | 老用户实时重排 |
| POST | `/api/cold_start` | 冷启动推荐（含 alpha 演化） |
| GET | `/api/stats` | 模型评估指标 |

---

## Dataset License

KuaiRand-1K 数据集遵循其原始许可协议，详见 `KuaiRand-1K/LICENSE`。
