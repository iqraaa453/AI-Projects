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
