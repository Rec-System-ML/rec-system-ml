"""
gen_interest_data.py
---------------------
从 KuaiRand-1K 日志计算 interest_evolution.html 所需的 USERS_DATA。
运行: python gen_interest_data.py
输出: ui_prototype/interest_evolution_data.js  (可直接 <script src=> 引入)
"""
import re, json, os
import pandas as pd
from collections import defaultdict

# ── 配置 ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'KuaiRand-1K', 'data')
OUT_FILE = os.path.join(ROOT, 'ui_prototype', 'interest_evolution_data.js')

# 三个展示用户 (user_id, 活跃度标签)
TARGET_USERS = [
    (970,  'full_active'),
    (999,  'high_active'),
    (546,  'mutation_active'),
]

# 快照时间窗口 (YYYYMMDD 区间，左闭右闭)，最后一个是"预测"占位
WINDOWS = [
    (20220408, 20220413, '2022-04-08',  'COLD START',          '初始化兴趣图谱'),
    (20220414, 20220419, '2022-04-14',  'EARLY GROWTH',        '兴趣节点萌芽扩展'),
    (20220420, 20220425, '2022-04-20',  'INTEREST EXPANSION',  '兴趣图谱快速生长'),
    (20220426, 20220501, '2022-04-26',  'DEEP DIVE',           '核心兴趣强化聚焦'),
    (20220502, 20220508, '2022-05-02',  'STABILIZATION',       '兴趣结构趋于稳定'),
]

def tag_name(t):
    return f'Tag-{t}'

def parse_tag_ids(s):
    if pd.isna(s): return []
    return [int(x) for x in re.findall(r'\d+', str(s))]

# ── 加载数据 ─────────────────────────────────────────────────────────────────
print('Loading logs...')
df1 = pd.read_csv(os.path.join(DATA_DIR, 'log_standard_4_08_to_4_21_1k.csv'))
df2 = pd.read_csv(os.path.join(DATA_DIR, 'log_standard_4_22_to_5_08_1k.csv'))
df  = pd.concat([df1, df2], ignore_index=True)

print('Loading video features...')
vf = pd.read_csv(os.path.join(DATA_DIR, 'video_features_basic_1k.csv'),
                 usecols=['video_id', 'tag'])
vf['tag_list'] = vf['tag'].apply(parse_tag_ids)
tag_map = dict(zip(vf['video_id'], vf['tag_list']))  # vid -> [tag_id, ...]

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def first_tag(vid):
    tl = tag_map.get(vid, [])
    return tl[0] if tl else None

def compute_snapshot(uid_df, top_n=8, prev_top_tags=None, is_prediction=False):
    """
    给定一段时间窗口内的用户日志 uid_df，计算快照的 nodes + links。
    prev_top_tags: 上一期 top tag id set，用于突变检测
    """
    if uid_df.empty:
        return [], []

    clicked = uid_df[uid_df['is_click'] == 1].sort_values('time_ms')

    # ── 节点计算 ──
    tag_stats = defaultdict(lambda: {'clicks':0,'likes':0,'comments':0,'forwards':0,
                                      'play_ms':0,'dur_ms':0,'cnt':0})
    for _, row in clicked.iterrows():
        vid  = int(row['video_id'])
        tl   = tag_map.get(vid, [])
        if not tl: continue
        t    = tl[0]
        s    = tag_stats[t]
        s['clicks']   += 1
        s['likes']    += int(row.get('is_like', 0) or 0)
        s['comments'] += int(row.get('is_comment', 0) or 0)
        s['forwards'] += int(row.get('is_forward', 0) or 0)
        pm = row.get('play_time_ms', 0) or 0
        dm = row.get('duration_ms', 0) or 0
        if dm > 0:
            s['play_ms'] += pm
            s['dur_ms']  += dm
            s['cnt']     += 1

    if not tag_stats:
        return [], []

    total_clicks = sum(v['clicks'] for v in tag_stats.values()) or 1
    top_tags = sorted(tag_stats.items(), key=lambda x: -x[1]['clicks'])[:top_n]

    # 半径：最多 r=44，最少 r=10，按点击数比例
    max_c = max(v['clicks'] for _, v in top_tags) or 1
    prev_set = set(prev_top_tags) if prev_top_tags else set()

    nodes = []
    mutation_found = False
    mutation_tag   = None
    for tag_id, stat in top_tags:
        c   = stat['clicks']
        r   = int(10 + 34 * (c / max_c))
        eng = int((stat['likes'] + stat['comments'] + stat['forwards']) / c * 100) if c else 0
        ctr = int(c / total_clicks * 100)
        watch = int(stat['play_ms'] / stat['dur_ms'] * 100) if stat['dur_ms'] > 0 else 0
        watch = min(watch, 100)

        # 突变检测：前一期不在 top，这期突然超过 10% 份额
        share = c / total_clicks
        if prev_set and tag_id not in prev_set and share > 0.10 and not mutation_found:
            state = 'mutation'
            mutation_found = True
            mutation_tag   = tag_id
        else:
            state = 'active'

        nodes.append({
            'id':    tag_name(tag_id),
            'r':     r,
            'state': state,
            'eng':   eng,
            'ctr':   ctr,
            'watch': watch,
            '_tag_id': tag_id,
        })

    # ── 转移链接计算 ──
    # 连续两次点击的 tag 序列 → Markov 转移
    tag_seq = []
    for _, row in clicked.iterrows():
        t = first_tag(int(row['video_id']))
        if t is not None:
            tag_seq.append(t)

    top_tag_set = {n['_tag_id'] for n in nodes}
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(tag_seq) - 1):
        s_tag, t_tag = tag_seq[i], tag_seq[i+1]
        if s_tag in top_tag_set and t_tag in top_tag_set and s_tag != t_tag:
            trans[s_tag][t_tag] += 1

    links = []
    for s_tag, targets in trans.items():
        total_out = sum(targets.values()) or 1
        for t_tag, cnt in sorted(targets.items(), key=lambda x:-x[1])[:3]:
            p = round(cnt / total_out, 2)
            if p >= 0.12:
                links.append({'s': tag_name(s_tag), 't': tag_name(t_tag), 'p': p})

    # 清理内部字段
    for n in nodes:
        del n['_tag_id']

    return nodes, links, (mutation_found, mutation_tag)


def compute_prediction(last_nodes, last_links, prev_nodes):
    """
    基于最后一期的 Markov 链预测下一期：
    - active 节点保留，状态不变
    - 上一期 predicted 节点在此期也保留
    - 新增 1-2 个 predicted 节点（通过 Markov 一步传播找到还不在图中的节点）
    """
    # 复制最后一期节点
    nodes = [{**n} for n in last_nodes]
    existing = {n['id'] for n in nodes}

    # 通过链接关系找到邻居节点（已存在的）中概率高的目标
    target_scores = defaultdict(float)
    for lk in last_links:
        target_scores[lk['t']] += lk['p']

    # 找出不在当前图中、可以作为预测节点的 tag name
    # 简单策略：高频 tag ID 中还没出现的前几个
    all_known = [f'Tag-{t}' for t in [34,3,9,2,12,11,39,28,6,25,19,8,13,1,20,7,15,18,68,67]]
    candidates = [n for n in all_known if n not in existing][:5]

    # 添加 1-2 个 predicted 节点
    for name in candidates[:2]:
        nodes.append({'id': name, 'r': 16, 'state': 'predicted', 'eng': 0, 'ctr': 0, 'watch': 0})

    # 链接：沿用最后一期的链接，并给 predicted 节点加虚拟链接
    links = list(last_links)
    pred_nodes = [n for n in nodes if n['state'] == 'predicted']
    active_nodes = [n for n in nodes if n['state'] == 'active']
    if pred_nodes and active_nodes:
        # 取最大活跃节点 → 第一个预测节点
        top_active = max(active_nodes, key=lambda x: x['r'])
        links.append({'s': top_active['id'], 't': pred_nodes[0]['id'], 'p': round(0.15 + 0.1 * len(pred_nodes), 2)})
        if len(pred_nodes) > 1 and len(active_nodes) > 1:
            second_active = sorted(active_nodes, key=lambda x: -x['r'])[1]
            links.append({'s': second_active['id'], 't': pred_nodes[1]['id'], 'p': 0.18})

    return nodes, links


# ── 主计算 ────────────────────────────────────────────────────────────────────

PHASE_LABELS = [
    ('COLD START',        'Interest Graph Init',       'LOW',    'REPLAY'),
    ('EARLY GROWTH',      'Interest Nodes Emerging',   'LOW',    'REPLAY'),
    ('INTEREST EXPANSION','Interest Graph Expanding',  'MEDIUM', 'REPLAY'),
    ('DEEP DIVE',         'Core Interests Reinforced', 'MEDIUM', 'REPLAY'),
    ('STABILIZATION',     'Interest Structure Stable', 'HIGH',   'REPLAY'),
]

users_data = []

for uid, activity in TARGET_USERS:
    print(f'\nComputing user {uid} ({activity})...')
    user_df = df[df['user_id'] == uid]
    snapshots = []
    prev_top_tags = None

    for i, (d_start, d_end, date_str, _, _) in enumerate(WINDOWS):
        window_df = user_df[(user_df['date'] >= d_start) & (user_df['date'] <= d_end)]
        label, sub, evo_pct, status = PHASE_LABELS[i]

        result = compute_snapshot(window_df, top_n=8, prev_top_tags=prev_top_tags)
        nodes, links, (mut_found, mut_tag) = result

        # 如果点击太少，复用上一期
        if len(nodes) < 2 and snapshots:
            prev = snapshots[-1]
            nodes = [{**n} for n in prev['nodes']]
            links = list(prev['links'])
            mut_found = False

        if mut_found:
            label  = '⚠ MUTATION DETECTED'
            sub    = 'Anomalous Interest Shift · High Confidence'
            evo_pct= 'HIGH'
            status = 'ALERT'

        snap = {
            't':        i,
            'date':     date_str,
            'label':    label,
            'sub':      sub,
            'evoPct':   evo_pct,
            'status':   status,
            'nodes':    nodes,
            'links':    links,
        }
        if mut_found:
            snap['mutation'] = True

        snapshots.append(snap)
        prev_top_tags = [
            n.get('_tag_id') or (
                int(n['id'].split('-')[1]) if n['id'].startswith('Tag-') else None
            )
            for n in nodes
        ]
        print(f'  t={i} ({date_str}): {len(nodes)} nodes, {len(links)} links'
              + (' [MUTATION]' if mut_found else ''))

    # 预测帧
    if snapshots:
        last = snapshots[-1]
        second_last = snapshots[-2] if len(snapshots) >= 2 else last
        pred_nodes, pred_links = compute_prediction(last['nodes'], last['links'], second_last['nodes'])
        snapshots.append({
            't':      5,
            'date':   '→ PREDICTED',
            'label':  'FUTURE PREDICTION',
            'sub':    'Predicted Interest Evolution Path',
            'evoPct': 'HIGH',
            'status': 'PREDICTING',
            'nodes':  pred_nodes,
            'links':  pred_links,
        })
        print(f'  t=5 (prediction): {len(pred_nodes)} nodes, {len(pred_links)} links')

    users_data.append({
        '_uid':     uid,
        '_activity': activity,
        'snapshots': snapshots,
    })

# ── 生成 JS 输出 ──────────────────────────────────────────────────────────────

def to_js(obj, indent=0):
    """把 Python 对象转成紧凑 JS 字面量（不带引号的键名）。"""
    pad  = '  ' * indent
    pad1 = '  ' * (indent + 1)

    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if k.startswith('_'): continue   # 跳过内部字段
            items.append(f'{pad1}{k}:{to_js(v, indent+1)}')
        return '{\n' + ',\n'.join(items) + '\n' + pad + '}'
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        # 短列表单行
        inner = [to_js(x, indent+1) for x in obj]
        oneline = '[' + ', '.join(inner) + ']'
        if len(oneline) < 120 and '\n' not in oneline:
            return oneline
        return '[\n' + ',\n'.join(pad1 + to_js(x, indent+1) for x in obj) + '\n' + pad + ']'
    elif isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, bool):
        return 'true' if obj else 'false'
    elif isinstance(obj, float):
        return str(round(obj, 3))
    else:
        return str(obj)

uid_labels = {
    970: f'#970 · full_active',
    999: f'#999 · high_active',
    546: f'#546 · mutation_active',
}

js_lines = [
    '// AUTO-GENERATED by gen_interest_data.py — DO NOT EDIT MANUALLY',
    'const USERS_DATA = [',
]
for ud in users_data:
    uid = ud['_uid']
    act = ud['_activity']
    js_lines.append(f'  // ── User #{uid} ({act}) ──')
    js_lines.append('  ' + to_js({'snapshots': ud['snapshots']}, 1) + ',')
js_lines.append(']')

# 更新 userSel 选项（同时输出供参考）
js_lines.append('')
js_lines.append('// userSel option labels:')
for i, ud in enumerate(users_data):
    js_lines.append(f'// {i}: {uid_labels[ud["_uid"]]}')

out = '\n'.join(js_lines)
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(out)

print(f'\nDone! Written to {OUT_FILE}')
print(f'File size: {len(out):,} chars')
