"""Build rl_dashboard.html from dash_data.json."""
import json
from pathlib import Path
from pathlib import Path

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL Dynamic Pricing Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#e6edf3;display:flex;min-height:100vh}
nav{width:200px;background:#161b22;border-right:1px solid #30363d;padding:16px 0;flex-shrink:0}
.logo{padding:0 16px 20px;font-weight:700;font-size:13px;color:#a78bfa;letter-spacing:.5px}
.nav-btn{display:block;width:100%;text-align:left;padding:9px 16px;background:none;border:none;color:#8b949e;font-size:12px;cursor:pointer;border-left:3px solid transparent}
.nav-btn:hover,.nav-btn.active{color:#e6edf3;background:#21262d;border-left-color:#7c3aed}
main{flex:1;overflow-y:auto;padding:20px 24px}
.sec{display:none}.sec.active{display:block}
.topbar{margin-bottom:18px}
.page-title{font-size:20px;font-weight:700}
.page-sub{font-size:12px;color:#8b949e;margin-top:4px}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.kpi{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.kpi-lbl{font-size:10px;color:#484f58;text-transform:uppercase;letter-spacing:.5px}
.kpi-val{font-size:22px;font-weight:700;font-family:monospace;margin:4px 0}
.kpi-note{font-size:10px;color:#8b949e}
.kpi.purple .kpi-val{color:#a78bfa}.kpi.green .kpi-val{color:#4ade80}
.kpi.amber .kpi-val{color:#f59e0b}.kpi.blue .kpi-val{color:#38bdf8}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:14px}
.card-t{font-size:13px;font-weight:600;margin-bottom:4px}
.card-s{font-size:11px;color:#8b949e;margin-bottom:12px}
.chart-wrap{position:relative;width:100%}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.legend{display:flex;gap:14px;margin-bottom:8px;font-size:11px;color:#8b949e}
.leg{display:flex;align-items:center;gap:5px}
.leg-sq{width:10px;height:10px;border-radius:2px}
.table{width:100%;border-collapse:collapse;font-size:12px}
.table th,.table td{padding:8px 10px;text-align:left;border-bottom:1px solid #21262d}
.table th{color:#8b949e;font-weight:500}
.badge{padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600}
.badge.p{background:rgba(124,58,237,.2);color:#a78bfa}
.badge.a{background:rgba(245,158,11,.2);color:#f59e0b}
.badge.r{background:rgba(248,113,113,.2);color:#f87171}
.act-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.act-item{background:#21262d;border-radius:6px;padding:10px;text-align:center}
.act-pct{font-size:18px;font-weight:700;font-family:monospace}
.act-lbl{font-size:10px;color:#8b949e;margin:2px 0}
.act-bar{height:3px;border-radius:2px;margin-top:6px}
.sim-grid{display:grid;grid-template-columns:280px 1fr;gap:14px}
.sim-ctrl label{font-size:11px;color:#8b949e}
select,input[type=range]{width:100%;margin:8px 0 14px;background:#21262d;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:8px}
.btn{width:100%;padding:10px;background:#7c3aed;color:#fff;border:none;border-radius:6px;font-weight:600;cursor:pointer}
.btn:hover{background:#6d28d9}
.sim-out{font-family:monospace;font-size:11px;color:#8b949e;line-height:1.8;max-height:200px;overflow-y:auto}
.sim-step{color:#e6edf3}
.ctrl-row{display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px}
.ctrl-lbl{color:#8b949e}.ctrl-val{font-family:monospace;color:#a78bfa}
.comp-row{display:grid;grid-template-columns:120px 70px 1fr 100px;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #21262d;font-size:12px}
.comp-name{font-weight:600}.comp-price{font-family:monospace;color:#e6edf3}
.comp-track{background:#21262d;border-radius:4px;height:6px;overflow:hidden}
.comp-fill{height:100%;border-radius:4px}
.comp-status{font-size:10px;padding:3px 8px;border-radius:4px;text-align:center}
.pill{display:flex;align-items:center;gap:6px;font-size:11px;color:#4ade80;background:rgba(74,222,128,.1);padding:6px 12px;border-radius:20px}
.dot-anim{width:8px;height:8px;background:#4ade80;border-radius:50%;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
</style>
</head>
<body>
<nav>
  <div class="logo">⚡ RL PRICING</div>
  <button class="nav-btn active" onclick="go('overview',this)">Overview</button>
  <button class="nav-btn" onclick="go('training',this)">Q-Learning</button>
  <button class="nav-btn" onclick="go('ddpg',this)">DDPG</button>
  <button class="nav-btn" onclick="go('comparison',this)">Comparison</button>
  <button class="nav-btn" onclick="go('sensitivity',this)">Sensitivity</button>
  <button class="nav-btn" onclick="go('simulator',this)">Simulator</button>
  <button class="nav-btn" onclick="go('competitor',this)">Competitors</button>
</nav>
<main>
<div id="overview" class="sec active">
  <div class="topbar"><div class="page-title">Overview</div><div class="page-sub">Dynamic pricing RL system · retail simulation</div></div>
  <div class="kpi-row">
    <div class="kpi purple"><div class="kpi-lbl">Best QL Profit</div><div class="kpi-val" id="k_best_ql">—</div><div class="kpi-note">Training peak</div></div>
    <div class="kpi green"><div class="kpi-lbl">QL Eval Mean</div><div id="k_ql_eval" class="kpi-val">—</div><div class="kpi-note">100 episodes</div></div>
    <div class="kpi amber"><div class="kpi-lbl">Rule-Based</div><div id="k_rule" class="kpi-val">—</div><div class="kpi-note">Heuristic baseline</div></div>
    <div class="kpi blue"><div class="kpi-lbl">Fixed Price</div><div id="k_fixed" class="kpi-val">—</div><div class="kpi-note">Always $100</div></div>
  </div>
  <div class="grid2">
    <div class="card"><div class="card-t">Strategy Comparison</div><div class="chart-wrap" style="height:200px"><canvas id="c_comp"></canvas></div></div>
    <div class="card"><div class="card-t">Q-Learning Training Curve</div><div class="card-s">Raw + 10-episode moving average</div><div class="chart-wrap" style="height:200px"><canvas id="c_learn"></canvas></div></div>
  </div>
  <div class="card"><div class="card-t">Action Distribution (Training)</div><div id="actGrid" class="act-grid"></div></div>
</div>
<div id="training" class="sec">
  <div class="topbar"><div class="page-title">Q-Learning Training</div><div class="page-sub">Discrete actions: −20%, −10%, 0%, +10%, +20%</div></div>
  <div class="grid2">
    <div class="card"><div class="card-t">Profit + Epsilon</div><div class="chart-wrap" style="height:220px"><canvas id="c_eps"></canvas></div></div>
    <div class="card"><div class="card-t">Average Price</div><div class="chart-wrap" style="height:220px"><canvas id="c_price"></canvas></div></div>
  </div>
</div>
<div id="ddpg" class="sec">
  <div class="topbar"><div class="page-title">DDPG Agent</div><div class="page-sub">Continuous multiplier ∈ [0.7, 1.5]</div></div>
  <div class="card"><div class="card-t">DDPG vs Q-Learning</div><div class="chart-wrap" style="height:240px"><canvas id="c_ddpg"></canvas></div></div>
  <div class="card"><div class="card-t">DDPG Avg Price</div><div class="chart-wrap" style="height:180px"><canvas id="c_ddpg_price"></canvas></div></div>
</div>
<div id="comparison" class="sec">
  <div class="topbar"><div class="page-title">Profitability Comparison</div><div class="page-sub">Fixed vs Rule vs Q-Learning vs DDPG</div></div>
  <div class="card"><div class="card-t">Mean Profit</div><div class="chart-wrap" style="height:200px"><canvas id="c_bar3"></canvas></div></div>
  <div class="grid2">
    <div class="card"><div class="card-t">Distribution</div><div class="chart-wrap" style="height:220px"><canvas id="c_box"></canvas></div></div>
    <div class="card"><div class="card-t">Episode Profits (50)</div><div class="chart-wrap" style="height:220px"><canvas id="c_line3"></canvas></div></div>
  </div>
  <div class="card"><table class="table"><tr><th>Strategy</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>vs Fixed</th></tr>
  <tr id="tr_fixed"></tr><tr id="tr_rule"></tr><tr id="tr_ql"></tr></table></div>
</div>
<div id="sensitivity" class="sec">
  <div class="topbar"><div class="page-title">Sensitivity Analysis</div></div>
  <div class="card"><div class="chart-wrap" style="height:220px"><canvas id="c_sens_bar"></canvas></div></div>
  <div class="card"><div class="chart-wrap" style="height:240px"><canvas id="c_sens_line"></canvas></div></div>
  <div class="card"><table class="table" id="sens_table"><tr><th>#</th><th>Config</th><th>α</th><th>γ</th><th>ε-decay</th><th>Mean</th><th>Std</th></tr></table></div>
</div>
<div id="simulator" class="sec">
  <div class="topbar"><div><div class="page-title">Live Simulator</div><div class="page-sub">Real env + trained agents (run dashboard_server.py)</div></div></div>
  <div class="sim-grid">
    <div class="card sim-ctrl">
      <div class="card-t">Controls</div>
      <label>Strategy</label><select id="sim_strat"><option value="fixed">Fixed ($100)</option><option value="rule">Rule-Based</option><option value="ql">Q-Learning (trained)</option><option value="random">Random</option></select>
      <div class="ctrl-row"><span class="ctrl-lbl">Steps</span><span class="ctrl-val" id="sim_steps_v">120</span></div>
      <input type="range" id="sim_steps" min="24" max="720" step="24" value="120" oninput="document.getElementById('sim_steps_v').textContent=this.value">
      <button class="btn" onclick="runSim()">Run Simulation</button>
      <div id="sim_kpis" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px"></div>
    </div>
    <div>
      <div class="card"><div class="sim-out" id="sim_out">Click Run Simulation...</div></div>
      <div class="card"><div class="chart-wrap" style="height:160px"><canvas id="c_sim"></canvas></div></div>
    </div>
  </div>
</div>
<div id="competitor" class="sec">
  <div class="topbar"><div class="page-title">Competitor Tracker</div><div class="pill"><div class="dot-anim"></div>Live</div></div>
  <div class="kpi-row">
    <div class="kpi purple"><div class="kpi-lbl">Our Price</div><div class="kpi-val" id="our_price">$100</div></div>
    <div class="kpi green"><div class="kpi-lbl">Market Avg</div><div class="kpi-val" id="mkt_avg">—</div></div>
    <div class="kpi amber"><div class="kpi-lbl">Cheapest</div><div class="kpi-val" id="mkt_min">—</div></div>
    <div class="kpi blue"><div class="kpi-lbl">Position</div><div class="kpi-val" id="mkt_pos">—</div></div>
  </div>
  <div class="card"><div id="comp_rows"></div></div>
  <div class="card"><div class="chart-wrap" style="height:180px"><canvas id="c_comp_hist"></canvas></div></div>
</div>
</main>
<script>
const D = __DATA__;
const fmt = v => '$'+Math.round(v).toLocaleString();
const colors = ['#a78bfa','#38bdf8','#4ade80','#f59e0b','#f87171','#fb923c','#e879f9'];
const chartsBuilt = {};
function go(id, btn) {
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  if (!chartsBuilt[id]) { buildCharts(id); chartsBuilt[id]=true; }
  if (id==='competitor' && !window._compStarted) { startCompetitor(); window._compStarted=true; }
}
function chartDefaults() {
  return { responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{backgroundColor:'#161b22',borderColor:'#30363d',callbacks:{label:c=>fmt(c.parsed.y)}}},
    scales:{ x:{grid:{color:'#21262d'},ticks:{color:'#484f58',maxTicksLimit:8}},
             y:{grid:{color:'#21262d'},ticks:{color:'#484f58',callback:v=>fmt(v)}} } };
}
function buildCharts(sec) {
  if (sec==='overview') buildOverview();
  else if (sec==='training') buildTraining();
  else if (sec==='ddpg') buildDDPG();
  else if (sec==='comparison') buildComparison();
  else if (sec==='sensitivity') buildSensitivity();
}
function buildOverview() {
  document.getElementById('k_best_ql').textContent = fmt(Math.max(...D.ql_profits));
  document.getElementById('k_ql_eval').textContent = fmt(D.comp_ql_mean);
  document.getElementById('k_rule').textContent = fmt(D.comp_rule_mean);
  document.getElementById('k_fixed').textContent = fmt(D.comp_fixed_mean);
  const labels=['-20%','-10%','0%','+10%','+20%'];
  const clrs=['#38bdf8','#4ade80','#f59e0b','#f87171','#a78bfa'];
  const total=D.action_dist.reduce((a,b)=>a+b,0);
  document.getElementById('actGrid').innerHTML = D.action_dist.map((v,i)=>{
    const pct=(v/total*100).toFixed(1);
    return `<div class="act-item"><div class="act-pct" style="color:${clrs[i]}">${pct}%</div><div class="act-lbl">${labels[i]}</div><div class="act-bar" style="background:${clrs[i]};width:${Math.round(v/total*100)}%"></div></div>`;
  }).join('');
  new Chart(document.getElementById('c_comp'),{type:'bar',data:{labels:['Fixed','Rule','QL','DDPG'],datasets:[{data:[D.comp_fixed_mean,D.comp_rule_mean,D.comp_ql_mean,D.comp_ddpg_mean||0],backgroundColor:['rgba(59,130,246,.6)','rgba(245,158,11,.6)','rgba(124,58,237,.7)','rgba(20,184,166,.7)'],borderRadius:4}]},options:chartDefaults()});
  const eps=Array.from({length:D.ql_profits.length},(_,i)=>i+1);
  new Chart(document.getElementById('c_learn'),{type:'line',data:{labels:eps,datasets:[
    {data:D.ql_profits,borderColor:'rgba(124,58,237,.3)',borderWidth:1,pointRadius:0,tension:.3},
    {data:D.ql_smooth,borderColor:'#a78bfa',borderWidth:2,pointRadius:0,tension:.4,fill:true,backgroundColor:'rgba(167,139,250,.08)'}
  ]},options:chartDefaults()});
}
function buildTraining() {
  const eps=Array.from({length:D.ql_profits.length},(_,i)=>i+1);
  new Chart(document.getElementById('c_eps'),{type:'line',data:{labels:eps,datasets:[
    {data:D.ql_smooth,borderColor:'#a78bfa',yAxisID:'y',pointRadius:0,tension:.4},
    {data:D.ql_eps,borderColor:'#f59e0b',yAxisID:'y1',pointRadius:0,borderDash:[4,2]}
  ]},options:{...chartDefaults(),scales:{x:{grid:{color:'#21262d'}},y:{position:'left',ticks:{callback:v=>fmt(v)}},y1:{position:'right',grid:{drawOnChartArea:false},ticks:{color:'#f59e0b'}}}}});
  new Chart(document.getElementById('c_price'),{type:'line',data:{labels:eps,datasets:[
    {data:D.ql_prices,borderColor:'#4ade80',pointRadius:0,tension:.3},
    {data:eps.map(()=>100),borderColor:'#30363d',borderDash:[3,3],pointRadius:0}
  ]},options:{...chartDefaults(),scales:{y:{ticks:{callback:v=>'$'+v}}}}});
}
function buildDDPG() {
  const n=Math.max(D.ql_smooth.length,D.ddpg_smooth.length);
  const eps=Array.from({length:n},(_,i)=>i+1);
  new Chart(document.getElementById('c_ddpg'),{type:'line',data:{labels:eps,datasets:[
    {data:D.ql_smooth,borderColor:'#a78bfa',pointRadius:0,tension:.4},
    {data:D.ddpg_smooth,borderColor:'#38bdf8',pointRadius:0,tension:.4,borderDash:[5,2]}
  ]},options:chartDefaults()});
  new Chart(document.getElementById('c_ddpg_price'),{type:'line',data:{labels:Array.from({length:D.ddpg_prices.length},(_,i)=>i+1),datasets:[{data:D.ddpg_prices,borderColor:'#38bdf8',pointRadius:0,tension:.4}]},options:chartDefaults()});
}
function buildComparison() {
  const strats=[{id:'tr_fixed',name:'Fixed',data:D.comp_fixed_mean,std:D.comp_fixed_std,all:D.comp_fixed_all,col:'#3b82f6'},
    {id:'tr_rule',name:'Rule-Based',data:D.comp_rule_mean,std:D.comp_rule_std,all:D.comp_rule_all,col:'#f59e0b'},
    {id:'tr_ql',name:'Q-Learning',data:D.comp_ql_mean,std:D.comp_ql_std,all:D.comp_ql_all,col:'#7c3aed'},
    {id:'tr_ddpg',name:'DDPG',data:D.comp_ddpg_mean||0,std:D.comp_ddpg_std||0,all:D.comp_ddpg_all||[],col:'#14b8a6'}];
  strats.forEach(s=>{
    const vf=(s.data-D.comp_fixed_mean)/D.comp_fixed_mean*100;
    document.getElementById(s.id).innerHTML=`<td><b>${s.name}</b></td><td style="color:${s.col};font-family:monospace">${fmt(s.data)}</td><td>±${Math.round(s.std)}</td><td>${fmt(Math.min(...s.all))}</td><td>${fmt(Math.max(...s.all))}</td><td>${vf>=0?'+':''}${vf.toFixed(1)}%</td>`;
  });
  new Chart(document.getElementById('c_bar3'),{type:'bar',data:{labels:['Fixed','Rule','QL','DDPG'],datasets:[{data:[D.comp_fixed_mean,D.comp_rule_mean,D.comp_ql_mean,D.comp_ddpg_mean||0],backgroundColor:['rgba(59,130,246,.6)','rgba(245,158,11,.6)','rgba(124,58,237,.7)','rgba(20,184,166,.7)'],borderRadius:4}]},options:chartDefaults()});
  const jitter=()=>(Math.random()-.5)*.3;
  new Chart(document.getElementById('c_box'),{type:'scatter',data:{datasets:[
    {data:D.comp_fixed_all.map(v=>({x:0+jitter(),y:v})),backgroundColor:'rgba(59,130,246,.4)',pointRadius:3},
    {data:D.comp_rule_all.map(v=>({x:1+jitter(),y:v})),backgroundColor:'rgba(245,158,11,.4)',pointRadius:3},
    {data:D.comp_ql_all.map(v=>({x:2+jitter(),y:v})),backgroundColor:'rgba(124,58,237,.4)',pointRadius:3},
    {data:(D.comp_ddpg_all||[]).map(v=>({x:3+jitter(),y:v})),backgroundColor:'rgba(20,184,166,.4)',pointRadius:3}
  ]},options:{...chartDefaults(),scales:{x:{min:-.5,max:2.5,ticks:{callback:v=>['Fixed','Rule','QL'][Math.round(v)]||''}}}}});
  const ep50=Array.from({length:50},(_,i)=>i+1);
  new Chart(document.getElementById('c_line3'),{type:'line',data:{labels:ep50,datasets:[
    {data:D.comp_fixed_all.slice(0,50),borderColor:'#3b82f6',pointRadius:0,tension:.3},
    {data:D.comp_rule_all.slice(0,50),borderColor:'#f59e0b',pointRadius:0,tension:.3},
    {data:D.comp_ql_all.slice(0,50),borderColor:'#a78bfa',pointRadius:0,tension:.3},
    {data:(D.comp_ddpg_all||[]).slice(0,50),borderColor:'#14b8a6',pointRadius:0,tension:.3}
  ]},options:chartDefaults()});
}
function buildSensitivity() {
  new Chart(document.getElementById('c_sens_bar'),{type:'bar',data:{labels:D.sens_labels,datasets:[{data:D.sens_means,backgroundColor:'rgba(124,58,237,.7)',borderRadius:4}]},options:chartDefaults()});
  if(D.ddpg_sens_labels) new Chart(document.getElementById('c_ddpg_sens_bar'),{type:'bar',data:{labels:D.ddpg_sens_labels,datasets:[{data:D.ddpg_sens_means,backgroundColor:'rgba(20,184,166,.7)',borderRadius:4}]},options:chartDefaults()});
  const ceps=Array.from({length:D.sens_curves[0].length},(_,i)=>i*3+1);
  new Chart(document.getElementById('c_sens_line'),{type:'line',data:{labels:ceps,datasets:D.sens_curves.map((c,i)=>({data:c,borderColor:colors[i],pointRadius:0,tension:.4}))},options:chartDefaults()});
  const sorted=D.sens_labels.map((l,i)=>({l,mean:D.sens_means[i],std:D.sens_stds[i],meta:D.sens_meta[i]})).sort((a,b)=>b.mean-a.mean);
  const tb=document.getElementById('sens_table');
  sorted.forEach((s,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${i+1}</td><td>${s.l}</td><td>${s.meta.alpha}</td><td>${s.meta.gamma}</td><td>${s.meta.decay}</td><td style="color:#a78bfa;font-family:monospace">${fmt(s.mean)}</td><td>±${Math.round(s.std)}</td>`;
    tb.appendChild(tr);
  });
}
// Live simulator + competitor — Python dashboard_server.py (real env + agents)
const API = window.location.origin;

async function api(path, opts) {
  const r = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function setApiStatus(ok, msg) {
  let el = document.getElementById('api_status');
  if (!el) {
    el = document.createElement('div');
    el.id = 'api_status';
    el.style.cssText =
      'position:fixed;bottom:12px;right:12px;padding:8px 12px;border-radius:6px;font-size:11px;z-index:99';
    document.body.appendChild(el);
  }
  el.style.background = ok ? 'rgba(74,222,128,.15)' : 'rgba(248,113,113,.15)';
  el.style.color = ok ? '#4ade80' : '#f87171';
  el.style.border = '1px solid ' + (ok ? '#4ade80' : '#f87171');
  el.textContent = msg;
}

async function checkApi() {
  try {
    const h = await api('/api/health');
    setApiStatus(true, 'Live API | Q-Learning loaded: ' + (h.ql_loaded ? 'yes' : 'no'));
    return true;
  } catch (e) {
    setApiStatus(false, 'Start server: python dashboard_server.py');
    return false;
  }
}

let simChart = null;

async function runSim() {
  const strat = document.getElementById('sim_strat').value;
  const steps = +document.getElementById('sim_steps').value;
  const out = document.getElementById('sim_out');
  out.textContent = 'Running live Python simulation...';
  try {
    const data = await api('/api/sim/run', {
      method: 'POST',
      body: JSON.stringify({ strategy: strat, steps }),
    });
    const fmt = v => '$' + Math.round(v).toLocaleString();
    out.innerHTML =
      data.log.map(l => `<span class="sim-step">${l}</span>`).join('<br>') +
      `<br><br><b style="color:#a78bfa">Total Profit: ${fmt(data.total_profit)}</b>` +
      `<br><span style="color:#8b949e">${data.strategy} | ${data.steps_run} steps | avg $${data.avg_price}</span>`;
    document.getElementById('sim_kpis').innerHTML = `
      <div style="background:#21262d;padding:8px;border-radius:6px;text-align:center">
        <div style="font-size:9px;color:#484f58">PROFIT</div>
        <div style="font-size:16px;font-weight:700;color:#a78bfa">${fmt(data.total_profit)}</div>
      </div>
      <div style="background:#21262d;padding:8px;border-radius:6px;text-align:center">
        <div style="font-size:9px;color:#484f58">AVG PRICE</div>
        <div style="font-size:16px;font-weight:700;color:#4ade80">$${data.avg_price}</div>
      </div>`;
    if (simChart) simChart.destroy();
    simChart = new Chart(document.getElementById('c_sim'), {
      type: 'line',
      data: {
        labels: data.prices.map((_, i) => i + 1),
        datasets: [
          { data: data.prices, borderColor: '#a78bfa', pointRadius: 0, tension: 0.3 },
          {
            data: data.demands,
            borderColor: '#4ade80',
            pointRadius: 0,
            tension: 0.3,
            borderDash: [4, 2],
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { position: 'left' }, y1: { position: 'right', grid: { drawOnChartArea: false } } },
      },
    });
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

let compChart = null;
let compInterval = null;
const compColors = ['#a78bfa', '#38bdf8', '#4ade80', '#f59e0b', '#f87171'];

async function updateCompetitor() {
  try {
    const data = await api('/api/market/tick');
    const ourP = data.our_price;
    document.getElementById('our_price').textContent = '$' + ourP;
    document.getElementById('mkt_avg').textContent = '$' + data.market_avg;
    document.getElementById('mkt_min').textContent = '$' + data.market_min;
    document.getElementById('mkt_pos').textContent =
      ourP < data.market_avg ? 'Below avg' : ourP > data.market_avg ? 'Above avg' : 'At parity';

    document.getElementById('comp_rows').innerHTML = data.peers
      .map((c, i) => {
        const pct = Math.round((c.price / 140) * 100);
        const st =
          c.price < ourP - 5
            ? '<span class="comp-status" style="background:rgba(248,113,113,.1);color:#f87171">Higher</span>'
            : c.price > ourP + 5
              ? '<span class="comp-status" style="background:rgba(74,222,128,.1);color:#4ade80">Cheaper</span>'
              : '<span class="comp-status" style="background:rgba(245,158,11,.1);color:#f59e0b">Parity</span>';
        return `<div class="comp-row">
          <div class="comp-name">${c.name}</div>
          <div class="comp-price">$${c.price.toFixed(1)}</div>
          <div class="comp-track"><div class="comp-fill" style="width:${pct}%;background:${compColors[i]}"></div></div>
          ${st}
        </div>`;
      })
      .join('');

    const hist = data.history;
    const ticks = hist.map((_, i) => i + 1);
    const ph = data.peer_history || {};
    if (compChart) compChart.destroy();
    compChart = new Chart(document.getElementById('c_comp_hist'), {
      type: 'line',
      data: {
        labels: ticks,
        datasets: [
          ...data.peers.map((c, i) => ({
            label: c.name,
            data: (ph[c.name] || []).slice(-ticks.length),
            borderColor: compColors[i],
            pointRadius: 0,
            tension: 0.4,
          })),
          {
            label: 'Our price (RL)',
            data: hist.map(h => h.our_price),
            borderColor: '#fff',
            borderDash: [4, 3],
            pointRadius: 0,
          },
          {
            label: 'Env competitor',
            data: hist.map(h => h.competitor_price),
            borderColor: '#f59e0b',
            borderDash: [2, 2],
            pointRadius: 0,
          },
        ],
      },
      options: chartDefaults(),
    });
  } catch (e) {
    console.warn('market tick failed', e);
  }
}

function startCompetitor() {
  if (compInterval) return;
  api('/api/market/reset', { method: 'POST', body: '{}' }).then(() => {
    updateCompetitor();
    compInterval = setInterval(updateCompetitor, 2000);
  });
}

checkApi();

buildCharts('overview'); chartsBuilt['overview']=true;
</script>
</body>
</html>"""


def _inject_live_js(template: str) -> str:
    live_path = Path(__file__).parent / "dashboard_live.js"
    if not live_path.exists():
        return template
    live = live_path.read_text(encoding="utf-8")
    start = template.find("// Live simulator + competitor")
    end = template.find("buildCharts('overview');")
    if start < 0 or end < 0:
        return template
    return template[:start] + live + "\n" + template[end:]


def main():
    with open("dash_data.json") as f:
        data = json.load(f)
    html = _inject_live_js(TEMPLATE.replace("__DATA__", json.dumps(data)))
    with open("rl_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved rl_dashboard.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()