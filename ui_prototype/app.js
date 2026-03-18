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
}
loadUserSelect()

// ═══════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════
const PAGE_LABELS = {
  dashboard: 'Dashboard',
  profile:   'User Profile',
  simulate:  'Live Simulate',
  visual:    'Interest Evo'
}
function goto(pageId, navEl) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'))
  document.querySelectorAll('.nav-link-item').forEach(n => n.classList.remove('active'))
  document.getElementById('page-' + pageId).classList.add('active')
  navEl.classList.add('active')
  document.getElementById('topbar-page').textContent = PAGE_LABELS[pageId]
  if (pageId === 'simulate') initSimulate()
  if (pageId === 'profile')  initProfileCharts()
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

  // KNN canvas
  drawKNN(42)
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
  document.getElementById('u-days').textContent   = `${pct}%`

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
  // New user tab does not need user select dropdown
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
  oldExtraClicks  = []
  oldPrevRanks    = {}
  popularCache    = null
  const h = document.getElementById('newClickHistory')
  if (h) h.style.display = 'none'
  initCurrentTab()
}

function initCurrentTab() {
  if (simCurrentTab === 'new') initNewUser()
  else initOldUser()
}

// ── New User Cold-Start ──────────────────────────────────────────────────────────────
async function initNewUser() {
  newClickedVids   = []
  newClickedData   = []
  coldStartResults = []
  const h = document.getElementById('newClickHistory')
  if (h) h.style.display = 'none'
  setBadge('Loading…', 'green')
  updateNewProgress(0, 'Phase 0 · Popularity Fallback')
  await Promise.all([loadPopular('newPopularList'), fetchColdStart()])
  setBadge('Click a video to start cold-start', 'purple')
}

let coldStartResults = []   // Cache latest cold-start results for click lookup
let newClickedData  = []   // Full data of clicked videos (for history panel)
let coldStartAlpha  = 0    // Current cold-start alpha (from API, grows with clicks)

async function fetchColdStart() {
  const data = await apiFetch('/cold-start/recommend', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({clicked_videos: newClickedVids, top_k: 10})
  })
  if (!data) return
  coldStartResults = data.results || []
  coldStartAlpha   = data.alpha || 0
  renderColdStartList('newColdList', coldStartResults, coldStartAlpha)
  updateNewProgress(data.progress_pct, data.phase_label)
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

function updateNewProgress(pct, phaseLabel) {
  const barPct = Math.round(pct * 100)
  document.getElementById('newProgressFill').style.width = barPct + '%'
  document.getElementById('newProgressText').textContent = barPct + '%'
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
// Score bar logic:
//   score = (1-α)×pop_score + α×knn_score  (backend formula, both normalized to [0,1])
//   orangeW = (1-α)×pop / score × 100       → Pop contribution ratio
//   purpleW = α×knn  / score × 100          → KNN contribution ratio
//   α=0 → all orange; α=1 → all purple; gradient in between
function renderColdStartList(id, list, alpha) {
  const el = document.getElementById(id)
  if (!el) return
  const a = alpha || 0
  el.innerHTML = (list || []).map((v, i) => {
    const tagStr     = (v.tags || []).slice(0, 2).join(' · ') || '--'
    const icon       = ICONS[i % ICONS.length]
    const popContrib = (1 - a) * (v.pop_score || 0)
    const knnContrib = a * (v.knn_score || 0)
    const total      = popContrib + knnContrib || 0.001
    const orangeW    = Math.round(popContrib / total * 100)
    const purpleW    = 100 - orangeW
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
        <div style="width:${orangeW}%;background:var(--orange);transition:width .4s ease"></div>
        <div style="width:${purpleW}%;background:var(--accent2);transition:width .4s ease"></div>
      </div>
      <div style="margin-top:2px;display:flex;justify-content:space-between;font-size:9px">
        <span style="color:var(--orange)">Pop ${orangeW}%</span>
        <span style="color:var(--accent2)">KNN ${purpleW}%</span>
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

// (Force graph removed — visual page now embeds interest_evolution.html via iframe)

// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════
window.addEventListener('load', () => {
  initProfileCharts()
})
