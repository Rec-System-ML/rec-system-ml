// ═══════════════════════════════════════
// API config
// ═══════════════════════════════════════
const API_BASE = 'http://localhost:8100/api'

// Generic fetch util (silent fallback when backend offline)
async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(API_BASE + path, options)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch (e) {
    console.warn('[API] Fallback to static data:', path, e.message)
    return null
  }
}

// ═══════════════════════════════════════
// Fetch KPI stats on startup
// ═══════════════════════════════════════
async function loadStats() {
  const data = await apiFetch('/stats')
  if (!data) return

  // KPI tiles
  const kpiMap = {
    'kpi-users':    data.user_count?.toLocaleString(),
    'kpi-videos':   (data.video_count / 1e6).toFixed(2) + 'M',
    'kpi-interact': (data.interaction_count / 1e6).toFixed(1) + 'M',
    'kpi-auc':      data.auc?.toFixed(4),
  }
  for (const [id, val] of Object.entries(kpiMap)) {
    const el = document.getElementById(id)
    if (el && val) el.textContent = val
  }

  // Model metrics cards
  const ndcg = data.ndcg_at_10 ?? 0.7636
  const auc  = data.auc ?? 0
  const prec = data.precision_at_10 ?? 0
  const rec  = data.recall_at_10 ?? 0.0596
  const cov  = data.catalog_coverage ?? 0.0086

  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val }
  const setW = (id, pct) => { const el = document.getElementById(id); if (el) el.style.width = pct + '%' }

  set('m-ndcg',   ndcg.toFixed(4))
  set('m-auc',    auc.toFixed(4))
  set('m-prec',   prec.toFixed(4))
  set('m-recall', rec.toFixed(4))
  set('m-cov',    (cov * 100).toFixed(2) + '%')
  setW('m-ndcg-bar', (ndcg * 100).toFixed(1))
  setW('m-auc-bar',  (auc  * 100).toFixed(1))
}
loadStats()

// Populate user select from API
async function loadUserSelect() {
  const data = await apiFetch('/users')
  if (!data?.users?.length) return
  const sel = document.getElementById('userSelect')
  if (!sel) return
  sel.innerHTML = data.users.slice(0, 100).map(uid =>
    `<option value="${uid}">User #${uid}</option>`
  ).join('')
  updateProfile(data.users[0])
}
loadUserSelect()

// ═══════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════
const PAGE_LABELS = {
  dashboard:        'Dashboard',
  profile:          'User Profile',
  simulate:         'Live Simulate',
  'interest-graph': 'Interest Graph',
}
function goto(pageId, navEl) {
  // Stop graph animation if leaving Interest Graph page
  if (pageId !== 'interest-graph') _stopGraphAnimation()

  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'))
  document.querySelectorAll('.nav-link-item').forEach(n => n.classList.remove('active'))
  document.getElementById('page-' + pageId).classList.add('active')
  navEl.classList.add('active')
  document.getElementById('topbar-page').textContent = PAGE_LABELS[pageId] || pageId
  if (pageId === 'simulate')         initSimulate()
  if (pageId === 'profile')          initProfileCharts()
  if (pageId === 'interest-graph')   initIgPage()
}

// ═══════════════════════════════════════
// FEEDBACK CHART (Dashboard)
// ═══════════════════════════════════════
Chart.defaults.color = '#6A6A90'
Chart.defaults.borderColor = '#252545'
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif"

;(function initFeedbackChart() {
  const ctx = document.getElementById('feedbackChart').getContext('2d')
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['CTR', 'Long Watch Rate', 'Like Rate'],
      datasets: [
        {
          label: 'Random Exp. (Unbiased)',
          data: [17.4, 8.4, 0.56],
          backgroundColor: 'rgba(56,189,248,.6)',
          borderColor: '#38BDF8',
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: 'Std Rec Phase1',
          data: [42.6, 30.5, 0.52],
          backgroundColor: 'rgba(91,79,232,.7)',
          borderColor: '#5B4FE8',
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: 'Std Rec Phase2',
          data: [31.2, 22.1, 0.28],
          backgroundColor: 'rgba(232,121,249,.6)',
          borderColor: '#E879F9',
          borderWidth: 1,
          borderRadius: 5,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, padding: 16, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` } }
      },
      scales: {
        x: { grid: { color: '#252545' } },
        y: {
          grid: { color: '#252545' },
          ticks: { callback: v => v + '%' }
        }
      }
    }
  })
})()

// ═══════════════════════════════════════
// MODEL COMPARISON CHART (Dashboard)
// ═══════════════════════════════════════
;(function initModelCompareChart() {
  const ctx = document.getElementById('modelCompareChart').getContext('2d')
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Baseline (Random)', 'Baseline + ItemKNN', 'Full Model\n(+ XGBoost Blend)'],
      datasets: [
        {
          label: 'AUC',
          data: [0.500, 0.502, 0.640],
          backgroundColor: 'rgba(67,24,255,.75)',
          borderColor: '#4318FF',
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: 'Accuracy',
          data: [0.174, 0.318, 0.679],
          backgroundColor: 'rgba(1,181,116,.75)',
          borderColor: '#01B574',
          borderWidth: 1,
          borderRadius: 5,
        }
      ]
    },
    options: {
      responsive: true,
      barPercentage: 0.45,
      categoryPercentage: 0.6,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, padding: 16, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}` } }
      },
      scales: {
        x: { grid: { color: '#252545' } },
        y: {
          min: 0, max: 1,
          grid: { color: '#252545' },
          ticks: { callback: v => v.toFixed(1) }
        }
      }
    }
  })
})()

// ═══════════════════════════════════════
// PROFILE CHARTS
// ═══════════════════════════════════════
const TAG_COLORS_POOL = ['#4318FF','#21AEF3','#01B574','#FF9A3C','#868CFF','#FF3CAC','#f59e0b','#10b981']
let _tagChartInstance = null

function _renderTagLegend(items) {
  const legend = document.getElementById('tagLegend')
  legend.innerHTML = ''
  items.forEach((item, i) => {
    const color = TAG_COLORS_POOL[i % TAG_COLORS_POOL.length]
    const row = document.createElement('div')
    row.style.cssText = 'display:grid;grid-template-columns:10px 52px 1fr 34px;align-items:center;gap:8px;'
    row.innerHTML = `
      <span style="width:10px;height:10px;border-radius:50%;background:${color};display:inline-block;"></span>
      <span style="font-size:12px;color:#cbd5e1;">${item.name}</span>
      <div style="height:4px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden;">
        <div style="width:${item.pct}%;height:100%;background:${color};border-radius:4px;"></div>
      </div>
      <span style="font-size:12px;color:#64748b;text-align:right;">${item.pct}%</span>`
    legend.appendChild(row)
  })
}

// Update donut chart with API data (callable after initProfileCharts)
function updateTagChart(tagProfile) {
  const items  = tagProfile.slice(0, 8)
  const labels = items.map(t => t.name)
  const values = items.map(t => t.pct)
  const colors = items.map((_, i) => TAG_COLORS_POOL[i % TAG_COLORS_POOL.length])
  _renderTagLegend(items)
  document.getElementById('tagCenterVal').textContent = items.length
  if (_tagChartInstance) {
    _tagChartInstance.data.labels = labels
    _tagChartInstance.data.datasets[0].data = values
    _tagChartInstance.data.datasets[0].backgroundColor = colors
    _tagChartInstance.update()
  }
}

let profileChartsInited = false
function initProfileCharts() {
  if (profileChartsInited) return
  profileChartsInited = true

  // Tag donut (static init values, updateProfile will override via API)
  const TAG_LABELS = ['Entertainment', 'Lifestyle', 'Food', 'Gaming', 'Education']
  const TAG_VALUES = [38, 24, 18, 12, 8]
  const TAG_COLORS = TAG_COLORS_POOL.slice(0, 5)
  _renderTagLegend(TAG_LABELS.map((name, i) => ({name, pct: TAG_VALUES[i]})))

  const tagCtx = document.getElementById('tagChart').getContext('2d')
  _tagChartInstance = new Chart(tagCtx, {
    type: 'doughnut',
    data: {
      labels: TAG_LABELS,
      datasets: [{
        data: TAG_VALUES,
        backgroundColor: TAG_COLORS,
        borderColor: '#111C44',
        borderWidth: 3,
        borderRadius: 6,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: false,
      animation: { animateRotate: true, duration: 900 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(11,20,55,.9)',
          borderColor: 'rgba(134,140,255,.3)',
          borderWidth: 1,
          callbacks: { label: ctx => `  ${ctx.label}  ${ctx.parsed}%` }
        }
      },
      cutout: '74%'
    }
  })

  // Radar

  // KNN canvas + initial profile data
  updateProfile(document.getElementById('userSelect')?.value ?? 42)
}

function drawKNN(uid) {
  const canvas = document.getElementById('knnChart')
  const ctx = canvas.getContext('2d')
  const W = canvas.width = canvas.parentElement.offsetWidth - 40
  const H = canvas.height = 220
  ctx.clearRect(0,0,W,H)
  ctx.fillStyle = '#0B1437'
  ctx.fillRect(0,0,W,H)

  // Random points
  const pts = Array.from({length:40}, () => ({
    x: Math.random()*W, y: Math.random()*H,
    near: Math.random() < 0.25
  }))
  pts.forEach(p => {
    ctx.beginPath(); ctx.arc(p.x, p.y, p.near ? 5 : 3, 0, Math.PI*2)
    if (p.near) {
      ctx.fillStyle = '#5B4FE8'
      ctx.shadowColor = '#5B4FE8'; ctx.shadowBlur = 10
      ctx.fill()
      ctx.shadowBlur = 0
      ctx.beginPath()
      ctx.moveTo(W/2, H/2); ctx.lineTo(p.x, p.y)
      ctx.strokeStyle = 'rgba(91,79,232,.3)'; ctx.lineWidth = 1
      ctx.stroke()
    } else {
      ctx.fillStyle = '#252545'
      ctx.fill()
    }
    ctx.shadowBlur = 0
  })
  // Current user
  ctx.beginPath(); ctx.arc(W/2, H/2, 10, 0, Math.PI*2)
  ctx.fillStyle = '#E879F9'
  ctx.shadowColor = '#E879F9'; ctx.shadowBlur = 20
  ctx.fill(); ctx.shadowBlur = 0
  ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'
  ctx.fillText(`User #${uid}`, W/2, H/2 + 24)
}

async function updateProfile(uid) {
  drawKNN(uid)
  const data = await apiFetch(`/user/${uid}/profile`)
  if (!data) return

  // Activity percentile → level label
  const pct = data.activity_percentile ?? 0
  const level = pct >= 80 ? 'full' : pct >= 60 ? 'high' : pct >= 40 ? 'mid' : 'low'
  document.getElementById('u-active').textContent = level
  document.getElementById('u-follow').textContent = data.click_count?.toLocaleString() ?? '--'
  document.getElementById('u-fans').textContent   = (data.history_sample?.length ?? 0)
  document.getElementById('u-days').textContent   = data.register_days ?? '--'

  // Update tag donut chart
  if (data.tag_profile?.length) {
    updateTagChart(data.tag_profile)
  }
}

// ═══════════════════════════════════════
// SIMULATE PAGE — Cold-Start + Old User Realtime Update
// ═══════════════════════════════════════
const ICONS = ['🎬','🎮','🍜','😂','🏋️','📱','🎵','🐱','🎤','🏀']

let simUserId      = 42
let simCurrentTab  = 'new'      // 'new' | 'old'
let newClickedVids = []         // New User Cold-Start click history
let oldExtraClicks = []         // Old user new clicks in session
let popularCache   = null       // Popular Top10 cache (shared by both tabs)

// ── User list ──────────────────────────────────────────────────────────────────
async function loadSimUserSelect() {
  // Initially hide dropdown (default tab is new user)
  const sel = document.getElementById('simUserSelect')
  if (sel) sel.style.display = 'none'

  const data = await apiFetch('/users')
  if (!data?.users?.length) return
  if (!sel) return
  sel.innerHTML = data.users.slice(0, 100).map(uid =>
    `<option value="${uid}">User #${uid}</option>`
  ).join('')
  simUserId = parseInt(data.users[0])
  initCurrentTab()
}
loadSimUserSelect()

function onSimUserChange(uid) {
  simUserId = parseInt(uid)
  resetSim()
}

// ── Tab switch ─────────────────────────────────────────────────────────────────
function switchSimTab(tab) {
  simCurrentTab = tab
  document.getElementById('simTab-new').style.display = tab === 'new' ? '' : 'none'
  document.getElementById('simTab-old').style.display = tab === 'old' ? '' : 'none'

  // New user tab does not need user select; old tab does
  const sel = document.getElementById('simUserSelect')
  if (sel) sel.style.display = tab === 'old' ? '' : 'none'

  const activeStyle   = 'padding:6px 18px;border-radius:20px;border:1px solid var(--accent);background:var(--accent);color:#fff;font-size:12px;font-weight:600;cursor:pointer'
  const inactiveStyle = 'padding:6px 18px;border-radius:20px;border:1px solid var(--border);background:transparent;color:var(--muted);font-size:12px;font-weight:600;cursor:pointer'
  document.getElementById('simTabBtn-new').style.cssText = tab === 'new' ? activeStyle : inactiveStyle
  document.getElementById('simTabBtn-old').style.cssText = tab === 'old' ? activeStyle : inactiveStyle
  initCurrentTab()
}

function resetSim() {
  newClickedVids  = []
  newClickedData  = []
  coldStartResults = []
  coldStartConf   = 0
  oldExtraClicks  = []
  oldPrevRanks    = {}
  popularCache    = null
  const h = document.getElementById('newClickHistory')
  if (h) h.style.display = 'none'
  initCurrentTab()
}

function initCurrentTab() {
  if (simCurrentTab === 'new')   initNewUser()
  else if (simCurrentTab === 'old')   initOldUser()
  else if (simCurrentTab === 'graph') initGraphTab()
}

// ── New User Cold-Start ──────────────────────────────────────────────────────────────
async function initNewUser() {
  newClickedVids   = []
  newClickedData   = []
  coldStartResults = []
  coldStartConf    = 0
  const h = document.getElementById('newClickHistory')
  if (h) h.style.display = 'none'
  setBadge('Loading…', 'green')
  updateNewProgress(0, 'Phase 0 · Popularity Fallback')
  await Promise.all([loadPopular('newPopularList'), fetchColdStart()])
  setBadge('Click a video to start cold-start', 'purple')
}

let coldStartResults = []   // Cache latest cold-start results for click lookup
let newClickedData  = []   // Full data of clicked videos (for history panel)
let coldStartConf   = 0    // Current cluster confidence (from API, grows with clicks)

async function fetchColdStart() {
  const data = await apiFetch('/cold-start/recommend', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({clicked_videos: newClickedVids, top_k: 10})
  })
  if (!data) return
  coldStartResults = data.results || []
  coldStartConf    = data.confidence || 0
  renderColdStartList('newColdList', coldStartResults, coldStartConf)
  updateNewProgress(data.progress_pct, data.phase_label, data.cluster_id)
}

async function onNewUserClick(vid, el) {
  flashEl(el)
  const vidInt = parseInt(vid)
  newClickedVids.push(vidInt)
  // Save video data for history panel
  const found = coldStartResults.find(v => v.video_id === vidInt)
  if (found) newClickedData.push(found)
  setBadge(`Clicked ${newClickedVids.length} `, 'purple')
  updateClickHistory()
  await fetchColdStart()
}

function updateNewProgress(pct, phaseLabel, clusterId) {
  const barPct = Math.round(pct * 100)
  document.getElementById('newProgressFill').style.width = barPct + '%'

  // Phase 1-3 显示 Cluster #N；Phase 0 显示百分比
  const rightText = (clusterId !== undefined && clusterId >= 0)
    ? `Cluster #${clusterId}`
    : barPct + '%'
  document.getElementById('newProgressText').textContent = rightText

  const lbl = document.getElementById('newPhaseLabel')
  if (lbl && phaseLabel) lbl.textContent = phaseLabel
}

// ── Old User Realtime Update ────────────────────────────────────────────────────────────
async function initOldUser() {
  oldExtraClicks = []
  oldPrevRanks   = {}
  setBadge('Loading…', 'green')
  document.getElementById('oldClickCount').textContent = '0'
  await Promise.all([loadPopular('oldPopularList'), fetchRealtimeRecommend()])
  setBadge('Click a video to update recommendations', 'purple')
}

let oldPrevRanks = {}   // {video_id: rank}, track previous rank for change calc

async function fetchRealtimeRecommend() {
  const data = await apiFetch('/recommend/realtime', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({user_id: simUserId, extra_clicks: oldExtraClicks, top_k: 10})
  })
  if (!data) return
  renderRealtimeList('oldPersonalList', data.results)
  // Update prevRanks to current ranking
  oldPrevRanks = {}
  data.results.forEach(v => { oldPrevRanks[v.video_id] = v.rank })
}

async function onOldUserClick(vid, el) {
  flashEl(el)
  oldExtraClicks.push(parseInt(vid))
  document.getElementById('oldClickCount').textContent = oldExtraClicks.length
  setBadge(`Clicked ${oldExtraClicks.length} `, 'purple')
  await fetchRealtimeRecommend()
}

// ── Pop Top10 (shared by both tabs, with cache) ──────────────────────────────
async function loadPopular(listId) {
  if (!popularCache) {
    const data = await apiFetch('/popular?n=10')
    if (!data?.results) return
    popularCache = data.results
  }
  renderVideoList(listId, popularCache, null)
}

// ── Generic list render (for Pop Top10, no score bar) ───────────────────────
function renderVideoList(id, list, clickHandler) {
  const el = document.getElementById(id)
  if (!el) return
  el.innerHTML = (list || []).map((v, i) => {
    const tagStr  = (v.tags || []).slice(0, 2).join(' · ') || '--'
    const score   = v.score != null ? v.score.toFixed(4) : (v.click_count != null ? v.click_count.toLocaleString() + ' ' : '')
    const icon    = ICONS[i % ICONS.length]
    const onclick = clickHandler ? `onclick="${clickHandler}(${v.video_id},this)"` : ''
    const cursor  = clickHandler ? 'cursor:pointer' : ''
    return `
    <div class="video-item" ${onclick} style="${cursor}">
      <div class="video-thumb">${icon}</div>
      <div style="flex:1;min-width:0;">
        <div style="font-weight:600;font-size:13px;">Video #${v.video_id}</div>
        <div style="color:var(--muted);font-size:11px;margin-top:2px">${tagStr}</div>
      </div>
      <div class="video-score">${score}</div>
    </div>`
  }).join('')
}

// ── Cold-start list (with score composition bar) ────────────────────────────
// Phase 0:   Pop(orange) 100%
// Phase 1:   Pop(orange) × (1-conf) | Cluster(teal) × conf
// Phase 2:   Cluster(teal 60%) | KNN(purple 40%)
// Phase 3:   Cluster(teal 40%) | KNN(purple 60%) fusion
function renderColdStartList(id, list, conf) {
  const el = document.getElementById(id)
  if (!el) return
  const c = conf || 0
  el.innerHTML = (list || []).map((v, i) => {
    const tagStr  = (v.tags || []).slice(0, 2).join(' · ') || '--'
    const icon    = ICONS[i % ICONS.length]
    const phase   = v.phase ?? 0

    let barHtml, labelHtml
    if (phase === 1) {
      // 显示系统级混合权重，而非 per-item 得分（per-item 两端0导致bar跳变）
      const oW  = Math.round((1 - c) * 100)
      const cW  = 100 - oW
      barHtml   = `<div style="width:${oW}%;background:var(--orange);transition:width .4s ease"></div>
                   <div style="width:${cW}%;background:#00bcd4;transition:width .4s ease"></div>`
      labelHtml = `<span style="color:var(--orange)">Pop ${oW}%</span>
                   <span style="color:#00bcd4">Cluster ${cW}%</span>`
    } else if (phase === 2) {
      // Cluster(teal 60%) + KNN(purple 40%)
      const cs   = v.cluster_score || 0
      const ks   = v.knn_score || 0
      const tot  = cs * 0.6 + ks * 0.4 || 0.001
      const cW   = Math.round(cs * 0.6 / tot * 100)
      const kW   = 100 - cW
      barHtml   = `<div style="width:${cW}%;background:#00bcd4;transition:width .4s ease"></div>
                   <div style="width:${kW}%;background:var(--accent2);transition:width .4s ease"></div>`
      labelHtml = `<span style="color:#00bcd4">Cluster ${cW}%</span>
                   <span style="color:var(--accent2)">KNN ${kW}%</span>`
    } else if (phase === 3) {
      // Cluster(teal 40%) + KNN(purple 60%) fusion
      const cs  = v.cluster_score || 0
      const ks  = v.knn_score || 0
      const tot = cs * 0.4 + ks * 0.6 || 0.001
      const cW  = Math.round(cs * 0.4 / tot * 100)
      const kW  = 100 - cW
      barHtml   = `<div style="width:${cW}%;background:#00bcd4;transition:width .4s ease"></div>
                   <div style="width:${kW}%;background:var(--accent2);transition:width .4s ease"></div>`
      labelHtml = `<span style="color:#00bcd4">Cluster ${cW}%</span>
                   <span style="color:var(--accent2)">KNN ${kW}%</span>`
    } else {
      // Phase 0: 纯热门
      barHtml   = `<div style="width:100%;background:var(--orange);transition:width .4s ease"></div>`
      labelHtml = `<span style="color:var(--orange)">Pop 100%</span>`
    }

    return `
    <div class="video-item" onclick="onNewUserClick(${v.video_id},this)"
         style="cursor:pointer;flex-direction:column;align-items:stretch;padding:9px 14px">
      <div style="display:flex;align-items:center;gap:10px">
        <div class="video-thumb">${icon}</div>
        <div style="flex:1;min-width:0">
          <div style="font-weight:600;font-size:13px">Video #${v.video_id}</div>
          <div style="color:var(--muted);font-size:11px;margin-top:1px">${tagStr}</div>
        </div>
        <div class="video-score">${(v.score||0).toFixed(4)}</div>
      </div>
      <div style="margin-top:6px;height:5px;border-radius:3px;overflow:hidden;background:rgba(255,255,255,.06);display:flex">
        ${barHtml}
      </div>
      <div style="margin-top:2px;display:flex;justify-content:space-between;font-size:9px">
        ${labelHtml}
      </div>
    </div>`
  }).join('')
}

// ── Old-user realtime list (with rank-change badge) ─────────────────────────
function renderRealtimeList(id, list) {
  const el = document.getElementById(id)
  if (!el) return
  el.innerHTML = (list || []).map((v, i) => {
    const tagStr = (v.tags || []).slice(0, 2).join(' · ') || '--'
    const icon   = ICONS[i % ICONS.length]
    const prevRank = oldPrevRanks[v.video_id]
    let badge
    if (prevRank === undefined) {
      badge = `<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:rgba(0,229,160,.18);color:var(--accent);font-weight:700">NEW</span>`
    } else {
      const delta = prevRank - v.rank
      if (delta === 0)     badge = `<span style="font-size:11px;color:var(--muted)">—</span>`
      else if (delta > 0)  badge = `<span style="font-size:10px;color:#4ade80;font-weight:800">↑${delta}</span>`
      else                 badge = `<span style="font-size:10px;color:#f87171;font-weight:800">↓${Math.abs(delta)}</span>`
    }
    return `
    <div class="video-item" onclick="onOldUserClick(${v.video_id},this)" style="cursor:pointer">
      <div style="min-width:28px;text-align:center">${badge}</div>
      <div class="video-thumb">${icon}</div>
      <div style="flex:1;min-width:0">
        <div style="font-weight:600;font-size:13px">Video #${v.video_id}</div>
        <div style="color:var(--muted);font-size:11px;margin-top:1px">${tagStr}</div>
      </div>
      <div class="video-score">${(v.score||0).toFixed(4)}</div>
    </div>`
  }).join('')
}

// ── C: Click History Panel ─────────────────────────────────────────────────────
function updateClickHistory() {
  const container = document.getElementById('newClickHistory')
  const items     = document.getElementById('newClickHistoryItems')
  if (!container || !items) return
  if (!newClickedData.length) { container.style.display = 'none'; return }
  container.style.display = ''
  items.innerHTML = newClickedData.map(v => {
    const tag = (v.tags || [])[0] || '--'
    return `<div style="padding:6px 10px;border-radius:8px;background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.2);font-size:11px;white-space:nowrap">
      <div style="font-weight:600;color:var(--text)">Video #${v.video_id}</div>
      <div style="color:var(--muted);font-size:10px">${tag}</div>
    </div>`
  }).join('')
}

function flashEl(el) {
  if (el) { el.classList.add('clicked'); setTimeout(() => el.classList.remove('clicked'), 900) }
}


function setBadge(text, cls) {
  const b = document.getElementById('simBadge')
  if (!b) return
  b.innerHTML = `<i class="bi bi-circle-fill me-1" style="font-size:7px"></i>${text}`
  b.className = `status-badge ${cls}`
}

// initSimulate called on page switch (kept for backward compat)
function initSimulate() { initCurrentTab() }


// ═══════════════════════════════════════════════════════════════════════════════
// INTEREST GRAPH PAGE  (Module 2 — sidebar nav item "C")
// ═══════════════════════════════════════════════════════════════════════════════

let igUserId = 42   // currently selected user for the IG page

// Populate IG user select from /api/users
async function loadIgUserSelect() {
  const data = await apiFetch('/users')
  if (!data?.users?.length) return
  const sel = document.getElementById('igUserSelect')
  if (!sel) return
  sel.innerHTML = data.users.slice(0, 100).map(uid =>
    `<option value="${uid}">User #${uid}</option>`
  ).join('')
  igUserId = parseInt(data.users[0])
}

function onIgUserChange(uid) {
  igUserId = parseInt(uid)
}

// Called when navigating to the Interest Graph page
function initIgPage() {
  loadIgUserSelect()
  setIgBadge('Select user and click Generate Graph', 'purple')
}

function setIgBadge(text, cls) {
  const b = document.getElementById('igBadge')
  if (!b) return
  b.innerHTML = `<i class="bi bi-circle-fill me-1" style="font-size:7px"></i>${text}`
  b.className = `status-badge ${cls}`
}

// Generate graph for IG page
async function generateIgGraph() {
  _stopGraphAnimation()

  const btn = document.getElementById('igGenBtn')
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="bi bi-hourglass-split me-1"></i>Building…' }
  setIgBadge('Building interest graph…', 'green')

  // Try real user first, fall back to demo
  let data = await apiFetch(`/interest-graph/${igUserId}`)
  if (!data || !data.nodes || data.nodes.length === 0) {
    console.warn('[IG] Real data unavailable, using demo graph.')
    data = await apiFetch('/interest-graph/demo')
  }

  if (btn) { btn.disabled = false; btn.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i>Generate Graph' }

  if (!data?.nodes) { setIgBadge('Error loading graph data', 'purple'); return }

  // Update KPI tiles
  const ns = data.nodes || []
  const ls = data.links || []
  const setKpi = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val }
  setKpi('ig-kpi-active', ns.filter(n => n.status === 'active').length)
  setKpi('ig-kpi-mut',    ns.filter(n => n.status === 'mutation').length)
  setKpi('ig-kpi-pred',   ns.filter(n => n.status === 'predicted').length)
  setKpi('ig-kpi-edges',  ls.length)

  // Hide placeholder
  const ph = document.getElementById('igPlaceholder')
  if (ph) ph.style.display = 'none'

  renderInterestGraph(data, 'igGraphContainer')
  const tag = data.fallback ? ' (demo fallback)' : ''
  setIgBadge(`Ready · ${ns.length} nodes · ${ls.length} edges${tag}`, 'green')
}

// ── Animation frame management (used by Interest Graph page) ──────────────────
let _graphAnimFrame   = null
let _graphIsAnimating = false

function _stopGraphAnimation() {
  _graphIsAnimating = false
  if (_graphAnimFrame) { cancelAnimationFrame(_graphAnimFrame); _graphAnimFrame = null }
}

// ── D3.js Force-directed Graph Renderer ──────────────────────────────────────
//
//  Faithfully ports the sci-fi visual style from interest_graph/template.html:
//    • Glowing SVG filters (green / purple / red / edge / dot glow)
//    • Radial gradients for node core bloom
//    • Orbital rings around each node (count by status)
//    • Curved edges with linear gradient stroke + probability badges
//    • Star/burst shape for mutation nodes
//    • Animated canvas: floating background particles + edge particle trail
//    • Node halo on texture canvas (additive blending)
//    • D3 force simulation with drag support
// ─────────────────────────────────────────────────────────────────────────────
function renderInterestGraph(graphData, containerId) {
  const container = document.getElementById(containerId)
  if (!container) return

  _stopGraphAnimation()
  container.innerHTML = ''

  const nodes = (graphData.nodes || []).map(d => ({...d, _oD: d.decay, _oS: d.status}))
  const links = (graphData.links || []).map(d => ({...d}))

  const W = container.offsetWidth  || 900
  const H = Math.max(container.offsetHeight > 200 ? container.offsetHeight - 4 : 0, 680)

  // ── Helper: node visual properties ──────────────────────────────────────────
  const nr = d => {
    // active / mutation / predicted: FIXED size — no decay-based shrink
    // Only fading nodes shrink (they're heading toward death)
    if (d.status === 'mutation')  return 52
    if (d.status === 'predicted') return 44
    if (d.status === 'active')    return 58
    return 22 + d.decay * 16   // fading: 22–38, shrinks gradually
  }
  const nFill = d => {
    if (d.status === 'mutation')  return 'rgba(140,20,20,.65)'
    if (d.status === 'predicted') return `rgba(60,30,110,${.35 + d.decay * .3})`
    if (d.status === 'active')    return `rgba(8,60,45,${.35 + d.decay * .35})`
    return `rgba(80,80,80,${.25 + d.decay * .35})`
  }
  const nStroke = d => {
    if (d.status === 'mutation')  return 'rgba(255,110,110,.95)'
    if (d.status === 'predicted') return 'rgba(200,160,255,.88)'
    if (d.status === 'active')    return 'rgba(110,255,200,.92)'
    return 'rgba(150,150,150,.5)'
  }
  const nFilter = d => {
    if (d.status === 'mutation')  return 'url(#ig-gl-r)'
    if (d.status === 'predicted') return 'url(#ig-gl-p)'
    if (d.status === 'active')    return 'url(#ig-gl-g)'
    return null
  }
  const probColor = d => {
    const sn = typeof d.source === 'object' ? d.source : nodes.find(n => n.id === d.source)
    const tn = typeof d.target === 'object' ? d.target : nodes.find(n => n.id === d.target)
    if ((sn && sn._oS === 'mutation')  || (tn && tn._oS === 'mutation'))  return 'red'
    if ((sn && sn._oS === 'predicted') || (tn && tn._oS === 'predicted')) return 'purple'
    if ((sn && sn._oS === 'active')    || (tn && tn._oS === 'active'))    return 'green'
    return 'gray'
  }
  const mutPath = (R, s = 14, ir = 0.62) => {
    const pts = [], iR = R * ir, st = Math.PI / s; let a = -Math.PI / 2
    for (let i = 0; i < s * 2; i++) {
      const r = i % 2 === 0 ? R : iR
      pts.push(`${Math.cos(a) * r},${Math.sin(a) * r}`)
      a += st
    }
    return `M ${pts.join(' L ')} Z`
  }
  const cPath = d => {
    const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y
    const mx = (sx+tx)/2, my = (sy+ty)/2, dx = tx-sx, dy = ty-sy
    const l = Math.sqrt(dx*dx+dy*dy) || 1, off = 20 + d.probability*30
    return `M${sx},${sy}Q${mx+(-dy/l)*off},${my+(dx/l)*off} ${tx},${ty}`
  }
  const cMid = d => {
    const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y
    const mx = (sx+tx)/2, my = (sy+ty)/2, dx = tx-sx, dy = ty-sy
    const l = Math.sqrt(dx*dx+dy*dy) || 1, off = (20+d.probability*30)*.5
    return { x: mx+(-dy/l)*off, y: my+(dx/l)*off }
  }

  // ── DOM structure ────────────────────────────────────────────────────────────
  // Background canvas (floating particles)
  const bgCanvas = document.createElement('canvas')
  bgCanvas.width = W; bgCanvas.height = H
  bgCanvas.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:1'
  container.style.position = 'relative'
  container.appendChild(bgCanvas)
  const bgC = bgCanvas.getContext('2d')

  // Texture canvas (node halos + edge particles)
  const texCanvas = document.createElement('canvas')
  texCanvas.width = W; texCanvas.height = H
  texCanvas.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:2'
  container.appendChild(texCanvas)
  const txC = texCanvas.getContext('2d')

  // SVG layer (graph elements)
  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svgEl.setAttribute('width', W); svgEl.setAttribute('height', H)
  svgEl.style.cssText = 'position:absolute;inset:0;z-index:3'
  container.appendChild(svgEl)

  // HUD overlay (top-right)
  const hud = document.createElement('div')
  const mutN = nodes.filter(n => n.status==='mutation').length
  const actN = nodes.filter(n => n.status==='active').length
  hud.style.cssText = `
    position:absolute;right:14px;top:10px;z-index:9;padding:7px 12px;
    border-radius:6px;border:1px solid rgba(130,180,255,.5);
    background:linear-gradient(180deg,rgba(10,20,36,.92),rgba(8,14,28,.75));
    font-size:10px;line-height:1.5;letter-spacing:.4px;text-transform:uppercase;
    color:rgba(200,220,255,.9);pointer-events:none`
  hud.innerHTML = `
    USER INTEREST EVOLUTION &bull; TIME DECAY ENCODING<br/>
    <span class="ig-hud-counts">DATA POINTS: <span style="color:#5eeaff">${nodes.length}</span>
    &nbsp;|&nbsp; MUTATION: <span style="color:#ff4a4a">${mutN}</span>
    &nbsp;|&nbsp; ACTIVE: <span style="color:#2cffaa">${actN}</span></span>`
  container.appendChild(hud)

  // Timeline bar (bottom)
  const tlDiv = document.createElement('div')
  tlDiv.style.cssText = 'position:absolute;left:6%;right:6%;bottom:8px;z-index:9'
  tlDiv.innerHTML = `
    <div style="position:relative;height:22px;display:flex;align-items:center">
      <span id="ig-tl-ind" style="position:absolute;top:-15px;left:50%;transform:translateX(-50%);
        font-size:8px;color:#8cc0f0;text-transform:uppercase;letter-spacing:.45px;pointer-events:none">NOW</span>
      <input type="range" id="ig-scrubber" min="0" max="100" value="0" step="1"
             style="-webkit-appearance:none;width:100%;height:7px;border-radius:99px;outline:none;
                    cursor:pointer;border:1px solid rgba(170,210,255,.22);
                    background:linear-gradient(90deg,#3a5a8f,#5f9dff 25%,#ba6dff 50%,#76f5ff 75%,#8f6dff);
                    box-shadow:0 0 16px rgba(120,190,255,.35)"/>
    </div>
    <div style="margin-top:3px;display:flex;justify-content:space-between;
                font-size:9px;color:#a0c4f0;letter-spacing:.45px;text-transform:uppercase">
      <span>PAST (3 MONTHS)</span><span>PRESENT</span><span>FUTURE PREDICTION</span>
    </div>`
  container.appendChild(tlDiv)

  // ── D3 SVG setup ─────────────────────────────────────────────────────────────
  const svg  = d3.select(svgEl)
  const defs = svg.append('defs')

  // Glow filters
  const addGlowFilter = (id, devs) => {
    const f = defs.append('filter').attr('id', id)
      .attr('x','-400%').attr('y','-400%').attr('width','800%').attr('height','800%')
    const names = ['fa','fb','fc','fd']
    devs.forEach((dev, i) => {
      f.append('feGaussianBlur').attr('in','SourceGraphic').attr('stdDeviation',dev).attr('result',names[i])
    })
    const merge = f.append('feMerge')
    ;[...names.slice(0,devs.length)].reverse().forEach(r => merge.append('feMergeNode').attr('in',r))
    merge.append('feMergeNode').attr('in','SourceGraphic')
  }
  addGlowFilter('ig-gl-g', [4,12,28,48])   // active  → green glow
  addGlowFilter('ig-gl-p', [3,10,24,42])   // predicted → purple glow
  addGlowFilter('ig-gl-r', [4,14,30,50])   // mutation → red glow

  const fEdge = defs.append('filter').attr('id','ig-gl-edge')
    .attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%')
  fEdge.append('feGaussianBlur').attr('in','SourceGraphic').attr('stdDeviation',3).attr('result','ea')
  const fEM = fEdge.append('feMerge')
  fEM.append('feMergeNode').attr('in','ea'); fEM.append('feMergeNode').attr('in','SourceGraphic')

  const fDot = defs.append('filter').attr('id','ig-gl-dot')
    .attr('x','-300%').attr('y','-300%').attr('width','600%').attr('height','600%')
  ;[4,12,22].forEach((dev,i) => {
    fDot.append('feGaussianBlur').attr('in','SourceGraphic').attr('stdDeviation',dev).attr('result','da_'[i]||'dc')
  })
  const fDM = fDot.append('feMerge')
  ;['dc','db','da'].forEach(r => { try{ fDM.append('feMergeNode').attr('in',r) }catch(e){} })
  fDM.append('feMergeNode').attr('in','SourceGraphic')

  // Radial gradients (node core bloom)
  const addRG = (id, stops) => {
    const g = defs.append('radialGradient').attr('id', id)
    stops.forEach(([off, col]) => g.append('stop').attr('offset',off).attr('stop-color',col))
  }
  addRG('ig-rg-green',  [['0%','#ffffff'],['20%','#d4ffe8'],['55%','#60ffb8'],['100%','rgba(40,255,170,0)']])
  addRG('ig-rg-purple', [['0%','#ffffff'],['22%','#e0d0ff'],['55%','#b080ff'],['100%','rgba(160,100,255,0)']])
  addRG('ig-rg-red',    [['0%','#ffffff'],['18%','#ffc0c0'],['50%','#ff5050'],['100%','rgba(255,50,50,0)']])

  // Per-edge linear gradients (update endpoints each tick for correct direction)
  const gIds = []
  links.forEach((lk, i) => {
    const id = `ig-eg${i}`; gIds.push(id)
    const pc = probColor(lk)
    const g = defs.append('linearGradient').attr('id',id).attr('gradientUnits','userSpaceOnUse')
    if (pc==='red')    { g.append('stop').attr('offset','0%').attr('stop-color','#ff7070'); g.append('stop').attr('offset','100%').attr('stop-color','#ff4040') }
    else if (pc==='purple') { g.append('stop').attr('offset','0%').attr('stop-color','#80ddff'); g.append('stop').attr('offset','100%').attr('stop-color','#d080ff') }
    else if (pc==='gray')   { g.append('stop').attr('offset','0%').attr('stop-color','#a0a0a0'); g.append('stop').attr('offset','100%').attr('stop-color','#707070') }
    else { g.append('stop').attr('offset','0%').attr('stop-color','#70d8ff'); g.append('stop').attr('offset','45%').attr('stop-color','#a0ffe8'); g.append('stop').attr('offset','100%').attr('stop-color','#2cffaa') }
  })

  const eLay  = svg.append('g')
  const elLay = svg.append('g')
  const nLay  = svg.append('g')

  // ── Curved edges ─────────────────────────────────────────────────────────────
  const linkSel = eLay.selectAll('path').data(links).join('path')
    .attr('fill','none').attr('stroke-linecap','round').attr('filter','url(#ig-gl-edge)')
    .attr('stroke',(_,i)=>`url(#${gIds[i]})`)
    .attr('stroke-width', d => 1.2 + d.probability*9)
    .attr('stroke-opacity', d => 0.2 + d.probability*0.72)

  // Probability badges (pill labels on edges)
  const badgeG = elLay.selectAll('g').data(links).join('g')
  badgeG.append('rect')
    .attr('fill','rgba(8,18,34,.82)').attr('rx',8).attr('ry',8)
    .attr('width',48).attr('height',18).attr('x',-24).attr('y',-12)
    .attr('stroke-width',.8)
    .attr('stroke', d => {
      const pc = probColor(d)
      return pc==='green'?'rgba(60,255,180,.6)':pc==='purple'?'rgba(190,140,255,.6)':pc==='red'?'rgba(255,120,120,.6)':'rgba(160,160,160,.6)'
    })
  badgeG.append('text')
    .attr('text-anchor','middle').attr('font-size',9).attr('font-weight',600)
    .attr('letter-spacing',.3).attr('dy',2).attr('pointer-events','none')
    .attr('fill', d => {
      const pc = probColor(d)
      return pc==='green'?'#8affcc':pc==='purple'?'#d8b8ff':pc==='red'?'#ffaaaa':'#c0c0c0'
    })
    .text(d => `P=${d.probability.toFixed(2)}`)

  // ── Nodes ─────────────────────────────────────────────────────────────────────
  const nodeSel = nLay.selectAll('g.node').data(nodes, d => d.id)
    .join(enter => {
      const g = enter.append('g').attr('class','node')
      // node-inner: scale animations live here; outer g = position only (tick)
      const inner = g.append('g').attr('class','node-inner').attr('transform','scale(0)')
      inner.each(function(d) {
        const r = d3.select(this), R = nr(d)
        // Orbital rings
        const ringCount = d.status==='active'?4 : d.status==='predicted'?3 : d.status==='mutation'?4 : 2
        const ringBase  = d.status==='mutation'?'rgba(255,80,80,' : d.status==='active'?'rgba(100,255,200,' : d.status==='predicted'?'rgba(180,140,255,' : 'rgba(140,140,140,'
        for (let i = 1; i <= ringCount; i++) {
          const op = Math.max(0.07, 0.5 - i*0.09)
          r.append('circle').attr('class','orbit').attr('data-ri', i)
            .attr('fill','none').attr('pointer-events','none')
            .attr('r', R*(1+i*0.32))
            .attr('stroke', ringBase + op + ')')
            .attr('stroke-width', 1.6 - 0.18*i)
            .attr('stroke-dasharray', d.status==='fading'?'3,5':'none')
            .attr('filter', i<=2?'url(#ig-gl-edge)':'none')
        }
        // Main shape (star for mutation, circle for others)
        if (d.status === 'mutation') {
          r.append('path').attr('class','node-shape mut-shape').attr('d', mutPath(R))
          r.append('text').attr('font-size',20).attr('font-weight',700).attr('fill','#ffd4d4')
            .attr('text-anchor','middle').attr('pointer-events','none').attr('dy',7).text('⚠')
          r.append('text').attr('font-size',9).attr('fill','#ffaaaa')
            .attr('text-anchor','middle').attr('pointer-events','none').attr('dy',R+20)
            .text('Abnormal Mutation · High Confidence')
        } else {
          r.append('circle').attr('class','node-shape')
        }
        // Center bloom — created for ALL nodes so fading nodes look correct
        // when shown in their birth-window active state.
        // Opacity is controlled by applyScrub transitions based on current status.
        if (d._oS === 'mutation') {
          r.append('circle').attr('class','bloom-outer').attr('pointer-events','none').attr('r',R*.6).attr('fill','url(#ig-rg-red)').attr('opacity',.55).attr('filter','url(#ig-gl-dot)')
          r.append('circle').attr('class','bloom-inner').attr('pointer-events','none').attr('r',R*.22).attr('fill','#ffe0e0').attr('opacity',.85).attr('filter','url(#ig-gl-dot)')
        } else {
          const grad = d._oS === 'predicted' ? 'url(#ig-rg-purple)' : 'url(#ig-rg-green)'
          r.append('circle').attr('class','bloom-outer').attr('pointer-events','none').attr('r',R*.8).attr('fill',grad).attr('opacity',0).attr('filter','url(#ig-gl-dot)')
          r.append('circle').attr('class','bloom-inner').attr('pointer-events','none').attr('r',R*.32).attr('fill','#ffffff').attr('opacity',0).attr('filter','url(#ig-gl-dot)')
        }
        // Predicted tag
        if (d._oS === 'predicted') {
          r.append('text').attr('font-size',9).attr('fill','#d0c4ff')
            .attr('text-anchor','middle').attr('opacity',.9).attr('pointer-events','none')
            .attr('dy', -R-14).text('Predicted Interest')
        }
        // Label
        r.append('text').attr('font-size',13).attr('fill','#eef6ff')
          .attr('text-anchor','middle').attr('pointer-events','none').attr('dy',5)
          .text(d.label)
        // Metrics (active nodes only)
        if (d.metrics) {
          Object.entries(d.metrics).forEach(([k,v],i) => {
            r.append('text').attr('font-size',8).attr('fill','#b8d8f8')
              .attr('text-anchor','middle').attr('opacity',.85).attr('pointer-events','none')
              .attr('dy', R+16+i*11).text(`${k}: ${v}`)
          })
        }
      })
      return g
    })

  // Apply styles to shapes
  nodeSel.selectAll('circle.node-shape')
    .attr('r', d=>nr(d)).attr('fill',d=>nFill(d)).attr('stroke',d=>nStroke(d))
    .attr('stroke-width', d=>d.status==='active'||d.status==='mutation'?2.5:1.5)
    .attr('opacity', d=>d.status==='mutation'?1:d.status==='predicted'?.92:d.status==='active'?.97:0.36+d.decay*.44)
    .attr('filter', d=>nFilter(d))
  nodeSel.selectAll('path.mut-shape')
    .attr('d',d=>mutPath(nr(d))).attr('fill',d=>nFill(d)).attr('stroke',d=>nStroke(d))
    .attr('stroke-width',2).attr('opacity',.98).attr('filter','url(#ig-gl-r)')

  // Drag
  nodeSel.call(d3.drag()
    .on('start', (e,d) => { if(!e.active) sim.alphaTarget(.28).restart(); d.fx=d.x; d.fy=d.y })
    .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y })
    .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null })
  )

  // ── Force simulation ──────────────────────────────────────────────────────────
  const sim = d3.forceSimulation(nodes)
    .force('link',      d3.forceLink(links).id(d=>d.id).distance(d=>140-d.probability*45))
    .force('charge',    d3.forceManyBody().strength(-580))
    .force('center',    d3.forceCenter(W/2, H/2-14))
    .force('collision', d3.forceCollide().radius(d=>nr(d)+32).iterations(3))

  sim.on('tick', () => {
    nodes.forEach(d => {
      const pad = nr(d) + 26
      d.x = Math.max(pad, Math.min(W-pad, d.x))
      d.y = Math.max(pad, Math.min(H-pad, d.y))
    })
    linkSel.attr('d', cPath)
    links.forEach((d,i) => {
      defs.select(`#${gIds[i]}`)
        .attr('x1', d.source.x).attr('y1', d.source.y)
        .attr('x2', d.target.x).attr('y2', d.target.y)
    })
    badgeG.attr('transform', d => { const m=cMid(d); return `translate(${m.x},${m.y})` })
    nodeSel.attr('transform', d => `translate(${d.x},${d.y})`)
  })

  // ── Timeline scrubber logic ───────────────────────────────────────────────────
  // Index 5 = t01×10 = 0.5 = center of bar → "NOW"
  // Left half (0–4): past;  Right half (6–9): future
  const tlLabels = [
    '3 MONTHS AGO','2 MONTHS AGO','1 MONTH AGO',
    '2 WEEKS AGO','1 WEEK AGO',
    'NOW',
    '1 WEEK LATER','2 WEEKS LATER','PREDICTED','FUTURE'
  ]

  // ── Timeline scrubber: 3-phase lifecycle animation ───────────────────────────
  //   BIRTH  → spring burst (elastic overshoot, 700ms)
  //   ACTIVE → continuous size/color transitions (480ms cubic)
  //   DEATH  → white flash expand → implode to zero (110ms + 300ms backIn)
  //   EDGES  → fade-draw on birth, retract on death
  const applyScrub = t01 => {
    // 1. Time indicator
    const tInd = container.querySelector('#ig-tl-ind')
    const li   = Math.min(tlLabels.length - 1, Math.floor(t01 * tlLabels.length))
    if (tInd) { tInd.textContent = tlLabels[li]; tInd.style.left = `${t01 * 100}%` }

    const isFirstRun = nodes.every(n => n._alive === undefined)

    // 2. Compute decay + alive state
    // Design:
    //   LEFT  half (t01 0→0.5): historical evolution — interests appear, decay, die
    //   AT    0.5  (NOW):       current state frozen
    //   RIGHT half (t01 0.5→1): NOW state frozen + predicted nodes spring in on top
    nodes.forEach(n => {
      const prevAlive = n._alive
      const nts = n.timestamp != null ? n.timestamp : 0.5

      if (n._oS === 'predicted') {
        // ── Predicted nodes: invisible before NOW, appear in right half ──────
        if (t01 <= 0.5) {
          n.decay  = 0.01
          n.status = 'predicted'
          n._alive = false
        } else {
          const fp = (t01 - 0.5) / 0.5          // 0→1 across right half
          n.decay  = Math.min(1.0, fp * 2.4)     // reaches full at fp≈0.42
          n.status = 'predicted'
          n._alive = fp > 0.04                   // birth fires just after NOW
        }
      } else {
        // ── Non-predicted nodes: evolve left half, FREEZE at NOW (cap t01=0.5) ──
        const tEff = Math.min(t01, 0.5)

        if (nts > tEff + 0.08) {
          n.decay  = 0.01
          n.status = 'fading'
          n._alive = false
        } else {
          const age = Math.max(0, tEff - nts)
          n.decay   = Math.max(0.05, Math.exp(-3.5 * age))

          // Birth window: newly appeared nodes look active/green
          if (age < 0.12 && n._oS !== 'mutation') {
            n.status = 'active'
          } else {
            n.status = n._oS
          }
          if ((n._oS === 'active' || n._oS === 'fading') && n.decay < 0.22)
            n.status = 'fading'

          n._alive = n.decay > 0.14
        }
      }

      // Lifecycle events (skip first render)
      n._justBorn = !isFirstRun && prevAlive === false && n._alive === true
      n._justDied = !isFirstRun && prevAlive === true  && n._alive === false
    })

    // 3. Edge visible state (used only by edgeOpacity/edgeWidth helpers below)
    links.forEach(l => {
      const sn  = typeof l.source === 'object' ? l.source : nodes.find(n => n.id === l.source)
      const tn  = typeof l.target === 'object' ? l.target : nodes.find(n => n.id === l.target)
      const lts = l.timestamp != null ? l.timestamp
                  : Math.max(sn?.timestamp ?? 0, tn?.timestamp ?? 0)
      l._visible = lts <= t01 + 0.08 && (sn?._alive ?? false) && (tn?._alive ?? false)
    })

    // 4. ── Node scale animations (outer g = position, inner g = scale) ─────────
    if (isFirstRun) {
      // First render: snap to correct scale with no animation
      nodeSel.select('g.node-inner')
        .attr('transform', d => d._alive ? 'scale(1)' : 'scale(0)')
    } else {
      // BIRTH: spring elastic burst 0 → 1 (overshoot then settle)
      nodeSel.filter(d => d._justBorn)
        .select('g.node-inner')
        .attr('transform', 'scale(0)')
        .transition().duration(700).ease(d3.easeElasticOut.period(0.42))
        .attr('transform', 'scale(1)')

      // DEATH: brief flash expand → implode to zero
      nodeSel.filter(d => d._justDied)
        .select('g.node-inner')
        .transition().duration(110).ease(d3.easeQuadOut)
        .attr('transform', 'scale(1.35)')
        .transition().duration(310).ease(d3.easeBackIn.overshoot(1.8))
        .attr('transform', 'scale(0)')

      // STABLE alive → ensure scale(1)
      nodeSel.filter(d => !d._justBorn && !d._justDied && d._alive)
        .select('g.node-inner')
        .transition().duration(220)
        .attr('transform', 'scale(1)')

      // STABLE dead → ensure scale(0)
      nodeSel.filter(d => !d._justBorn && !d._justDied && !d._alive)
        .select('g.node-inner')
        .attr('transform', 'scale(0)')
    }

    // 5. ── Node property transitions (size / color / glow) ────────────────────
    const dur = 300
    nodeSel.selectAll('circle.node-shape')
      .interrupt('np').transition('np').duration(dur).ease(d3.easeCubicInOut)
      .attr('r',            d => nr(d))
      .attr('fill',         d => nFill(d))
      .attr('stroke',       d => nStroke(d))
      .attr('stroke-width', d => d.status === 'active' || d.status === 'mutation' ? 2.5 : 1.5)
      .attr('opacity',      d =>
        d.status === 'mutation'  ? 1 :
        d.status === 'predicted' ? 0.92 :
        d.status === 'active'    ? 0.97 :
        0.36 + d.decay * 0.44)
      .attr('filter', d => nFilter(d))

    nodeSel.selectAll('path.mut-shape')
      .interrupt('np').transition('np').duration(dur).ease(d3.easeCubicInOut)
      .attr('d', d => mutPath(nr(d))).attr('opacity', 0.98)

    nodeSel.selectAll('circle.orbit')
      .interrupt('np').transition('np').duration(dur).ease(d3.easeCubicInOut)
      .attr('r', function(d) {
        const ri = +d3.select(this).attr('data-ri') || 1
        return nr(d) * (1 + ri * 0.32)
      })
      .attr('stroke', d =>
        d.status === 'mutation'  ? 'rgba(255,80,80,0.35)' :
        d.status === 'predicted' ? 'rgba(180,140,255,0.35)' :
        d.status === 'active'    ? 'rgba(100,255,200,0.35)' :
        'rgba(140,140,140,0.25)')
      .attr('stroke-dasharray', d => d.status === 'fading' ? '3,5' : 'none')
      .attr('opacity', d =>
        !d._alive ? 0 :
        d.status === 'fading' ? Math.max(0.05, d.decay * 0.55) :
        0.85)

    // Bloom: visible when active/predicted/mutation, hidden when fading
    const isGlowing = d => d._alive && (d.status === 'active' || d.status === 'predicted' || d.status === 'mutation')
    nodeSel.selectAll('circle.bloom-outer')
      .interrupt('np').transition('np').duration(dur).ease(d3.easeCubicInOut)
      .attr('r', d => nr(d) * (d.status === 'mutation' ? 0.6 : 0.8))
      .attr('opacity', d => isGlowing(d) ? (d.status === 'mutation' ? 0.55 : 0.5) : 0)
    nodeSel.selectAll('circle.bloom-inner')
      .interrupt('np').transition('np').duration(dur).ease(d3.easeCubicInOut)
      .attr('r', d => nr(d) * (d.status === 'mutation' ? 0.22 : 0.32))
      .attr('opacity', d => isGlowing(d) ? (d.status === 'mutation' ? 0.85 : 0.92) : 0)

    // 6. ── Edge lifecycle animations ─────────────────────────────────────────
    const edgeSrc   = d => typeof d.source === 'object' ? d.source : nodes.find(n => n.id === d.source)
    const edgeTgt   = d => typeof d.target === 'object' ? d.target : nodes.find(n => n.id === d.target)
    const edgeAlive = d => (edgeSrc(d)?._alive ?? false) && (edgeTgt(d)?._alive ?? false)

    // Named transition 'ef' — replaces any queued 'ef' transition on the same element,
    // preventing the drag-stacking bug where rapid scrub events queue up 480ms anims.
    linkSel
      .interrupt('ef')
      .transition('ef').duration(300).ease(d3.easeCubicOut)
      .attr('stroke-opacity', d => edgeAlive(d) ? 0.2 + d.probability * 0.72 : 0)
      .attr('stroke-width',   d => edgeAlive(d) ? 1.2 + d.probability * 9   : 0)
    badgeG
      .interrupt('ef')
      .transition('ef').duration(250)
      .attr('opacity', d => edgeAlive(d) ? 0.95 : 0)

    // 7. Live HUD counts
    const aliveN = nodes.filter(n => n._alive)
    const hudEl  = container.querySelector('.ig-hud-counts')
    if (hudEl) hudEl.innerHTML =
      `DATA POINTS: <span style="color:#5eeaff">${aliveN.length}</span>` +
      ` &nbsp;|&nbsp; MUTATION: <span style="color:#ff4a4a">${aliveN.filter(n=>n.status==='mutation').length}</span>` +
      ` &nbsp;|&nbsp; ACTIVE: <span style="color:#2cffaa">${aliveN.filter(n=>n.status==='active').length}</span>`

    // 8. Gentle sim nudge for repositioning (lower alpha = less jitter)
    sim.alpha(0.06).restart()
  }

  // Bind scrubber input event
  const scrubEl = container.querySelector('#ig-scrubber')
  if (scrubEl) {
    scrubEl.addEventListener('input', e => applyScrub(Number(e.target.value) / 100))
  }

  // Initial scrub position = past (0), user drags right to reveal interests
  applyScrub(0)

  // ── Background particles ──────────────────────────────────────────────────────
  const mkParticles = (n, rMin, rMax, aMin, aMax, sMin, sMax) =>
    d3.range(n).map(() => ({
      x: Math.random()*W, y: Math.random()*H,
      r: rMin + Math.random()*(rMax-rMin),
      a: aMin + Math.random()*(aMax-aMin),
      s: sMin + Math.random()*(sMax-sMin),
      h: Math.random()<.4?272:Math.random()<.7?185:155
    }))
  const bgParticles = [
    mkParticles(80,.3,.9,.06,.2,.02,.06),
    mkParticles(40,.9,2.2,.12,.35,.06,.16),
  ]

  // ── Edge particles ─────────────────────────────────────────────────────────────
  const edgeParts = []
  links.forEach((lk, i) => {
    const pc=probColor(lk), isMut=pc==='red'
    const col = isMut?'rgba(255,90,90,':pc==='purple'?'rgba(180,130,255,':pc==='gray'?'rgba(160,160,160,':'rgba(120,255,200,'
    for (let j=0; j < (isMut?16:8); j++) {
      edgeParts.push({ li:i, t:Math.random(), speed:.0015+Math.random()*.005,
        r: isMut?1.5+Math.random()*2:.8+Math.random()*1.5, color:col })
    }
  })

  // ── Animation loop ─────────────────────────────────────────────────────────────
  _graphIsAnimating = true
  const animate = ts => {
    if (!_graphIsAnimating) return

    // — Background particles —
    bgC.clearRect(0,0,W,H)
    bgC.globalCompositeOperation = 'lighter'
    for (const ly of bgParticles) {
      for (const p of ly) {
        p.y -= p.s
        if (p.y < -8) { p.y = H+8; p.x = Math.random()*W }
        const pulse = .55 + .45*Math.sin(ts*.0012 + p.x*.008)
        bgC.beginPath()
        bgC.fillStyle = `hsla(${p.h},80%,60%,${p.a*pulse})`
        bgC.arc(p.x, p.y, p.r, 0, Math.PI*2); bgC.fill()
      }
    }
    bgC.globalCompositeOperation = 'source-over'

    // — Node halos —
    txC.clearRect(0,0,W,H)
    txC.globalCompositeOperation = 'lighter'
    nodes.forEach(n => {
      if (typeof n.x !== 'number' || n.status==='fading' || !n._alive) return
      const R=nr(n), haloR=R*2.6
      const col = n.status==='mutation'?'255,60,60':n.status==='predicted'?'170,120,255':'60,255,170'
      const g = txC.createRadialGradient(n.x,n.y,R*.3,n.x,n.y,haloR)
      g.addColorStop(0,   `rgba(${col},${.3*n.decay})`)
      g.addColorStop(.4,  `rgba(${col},${.12*n.decay})`)
      g.addColorStop(1,   `rgba(${col},0)`)
      txC.beginPath(); txC.fillStyle=g; txC.arc(n.x,n.y,haloR,0,Math.PI*2); txC.fill()
    })
    txC.globalCompositeOperation = 'source-over'

    // — Edge particle trails —
    txC.save()
    txC.globalCompositeOperation = 'lighter'
    for (const ep of edgeParts) {
      const d = links[ep.li]
      if (!d.source.x) continue
      // Skip particles if either endpoint node is dead
      if (!(d.source._alive ?? true) || !(d.target._alive ?? true)) continue
      ep.t += ep.speed; if (ep.t>1) ep.t -= 1
      const t=ep.t, sx=d.source.x, sy=d.source.y, ex=d.target.x, ey=d.target.y
      const mx=(sx+ex)/2, my=(sy+ey)/2, dx=ex-sx, dy=ey-sy, l=Math.sqrt(dx*dx+dy*dy)||1
      const off=20+d.probability*30
      const cx=mx+(-dy/l)*off, cy=my+(dx/l)*off
      const u=1-t, px=u*u*sx+2*u*t*cx+t*t*ex, py=u*u*sy+2*u*t*cy+t*t*ey
      txC.shadowColor=ep.color+'0.6)'; txC.shadowBlur=6
      txC.beginPath(); txC.fillStyle=ep.color+(.45+Math.random()*.4)+')'; txC.arc(px,py,ep.r,0,Math.PI*2); txC.fill()
    }
    txC.shadowBlur=0; txC.restore()

    _graphAnimFrame = requestAnimationFrame(animate)
  }
  _graphAnimFrame = requestAnimationFrame(animate)
}


// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════
window.addEventListener('load', () => {
  initProfileCharts()
})
