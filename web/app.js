/**
 * NEXUS REVOPS // DECISION COCKPIT
 * Frontend Application Engine (ES Modules & Three.js 3D Draco Mascot)
 */

const STATE = {
  metadata: null,
  currentDeals: [],
  selectedDeal: null,
  activeTab: 'tab-action',
  filter: {
    mode: 'agent', // 'agent' ou 'macro'
    agent: '',
    region: 'Todas',
    manager: 'Todos',
    quadrant: 'Todos',
    stage: 'Todos',
    search: '',
    sortBy: 'lead_score',
    sortAsc: false,
    limit: 500,
    offset: 0
  },
  settings: {
    provider: 'openrouter',
    openrouterKey: '',
    openrouterModel: 'nousresearch/hermes-3-llama-3.1-70b',
    openaiKey: '',
    temperature: 0.2
  },
  charts: {
    scatter: null,
    quadrants: null
  },
  three3D: {
    mascot: null,
    drawer: null
  }
};

// ==============================================================================
// INICIALIZAÇÃO
// ==============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  loadStoredSettings();
  setupEventListeners();
  await loadMetadata();
  await refreshAllData();
  initDraco3DMascots();
});

// ==============================================================================
// GESTÃO DE CONFIGURAÇÕES (LOCALSTORAGE & OPENROUTER)
// ==============================================================================
function loadStoredSettings() {
  try {
    const raw = localStorage.getItem('nexus_revops_settings') || localStorage.getItem('nous_revops_settings');
    if (raw) {
      const parsed = JSON.parse(raw);
      STATE.settings = { ...STATE.settings, ...parsed };
    }
  } catch (e) {
    console.warn('Erro ao carregar configurações salvas:', e);
  }
  updateCopilotBadge();
  populateSettingsModal();
}

function saveSettings() {
  const provider = document.getElementById('cfg-provider').value;
  const orKey = document.getElementById('cfg-openrouter-key').value.trim();
  let orModel = document.getElementById('cfg-openrouter-model').value;
  if (orModel === 'custom') {
    orModel = document.getElementById('cfg-custom-model').value.trim() || 'nousresearch/hermes-3-llama-3.1-70b';
  }
  const oaKey = document.getElementById('cfg-openai-key').value.trim();
  const temp = parseFloat(document.getElementById('cfg-temp').value) || 0.2;

  STATE.settings = {
    provider,
    openrouterKey: orKey,
    openrouterModel: orModel,
    openaiKey: oaKey,
    temperature: temp
  };

  localStorage.setItem('nexus_revops_settings', JSON.stringify(STATE.settings));
  updateCopilotBadge();
  closeSettingsModal();

  const statusBox = document.getElementById('modal-status-text');
  if (statusBox) {
    statusBox.textContent = '[CONFIRMADO] Configuração neural salva com sucesso no navegador.';
  }
}

function updateCopilotBadge() {
  const badge = document.getElementById('copilot-provider-badge');
  if (!badge) return;

  if (STATE.settings.provider === 'openrouter') {
    if (STATE.settings.openrouterKey) {
      const modelShort = STATE.settings.openrouterModel.split('/').pop();
      badge.textContent = `DRACO AI // ${modelShort.toUpperCase()}`;
      badge.className = 'telemetry-value active-pulse';
    } else {
      badge.textContent = 'DRACO AI // CHAVE PENDENTE';
      badge.className = 'telemetry-value';
    }
  } else if (STATE.settings.provider === 'openai') {
    badge.textContent = STATE.settings.openaiKey ? 'DRACO AI // OPENAI GPT-4O' : 'DRACO AI // CHAVE PENDENTE';
    badge.className = STATE.settings.openaiKey ? 'telemetry-value active-pulse' : 'telemetry-value';
  } else {
    badge.textContent = 'DRACO AI // OFFLINE GROUNDED';
    badge.className = 'telemetry-value';
  }
}

function populateSettingsModal() {
  document.getElementById('cfg-provider').value = STATE.settings.provider;
  document.getElementById('cfg-openrouter-key').value = STATE.settings.openrouterKey || '';
  
  const modelSelect = document.getElementById('cfg-openrouter-model');
  const customInput = document.getElementById('cfg-custom-model');
  
  const standardModels = [
    'nousresearch/hermes-3-llama-3.1-70b',
    'nousresearch/deephermes-3-llama-3-8b-preview',
    'nousresearch/hermes-3-llama-3.1-405b',
    'openai/gpt-4o-mini',
    'anthropic/claude-3.5-sonnet'
  ];

  if (standardModels.includes(STATE.settings.openrouterModel)) {
    modelSelect.value = STATE.settings.openrouterModel;
    customInput.style.display = 'none';
  } else {
    modelSelect.value = 'custom';
    customInput.style.display = 'block';
    customInput.value = STATE.settings.openrouterModel || '';
  }

  document.getElementById('cfg-openai-key').value = STATE.settings.openaiKey || '';
  document.getElementById('cfg-temp').value = STATE.settings.temperature;
  document.getElementById('val-temp').textContent = STATE.settings.temperature.toFixed(2);

  handleProviderChange(STATE.settings.provider);
}

function handleProviderChange(provider) {
  const grpOrKey = document.getElementById('grp-openrouter-key');
  const grpOrModel = document.getElementById('grp-openrouter-model');
  const grpOaKey = document.getElementById('grp-openai-key');

  if (provider === 'openrouter') {
    grpOrKey.style.display = 'block';
    grpOrModel.style.display = 'block';
    grpOaKey.style.display = 'none';
  } else if (provider === 'openai') {
    grpOrKey.style.display = 'none';
    grpOrModel.style.display = 'none';
    grpOaKey.style.display = 'block';
  } else {
    grpOrKey.style.display = 'none';
    grpOrModel.style.display = 'none';
    grpOaKey.style.display = 'none';
  }
}

// ==============================================================================
// METADATA & FILTROS
// ==============================================================================
async function loadMetadata() {
  try {
    const res = await fetch('/api/metadata');
    const data = await res.json();
    STATE.metadata = data;

    // Vendedores
    const selectAgent = document.getElementById('select-agent');
    selectAgent.innerHTML = '';
    data.agents.forEach((agent) => {
      const opt = document.createElement('option');
      opt.value = agent;
      opt.textContent = agent;
      selectAgent.appendChild(opt);
    });

    if (data.agents.length > 0) {
      STATE.filter.agent = data.agents[0];
      selectAgent.value = data.agents[0];
    }

    // Regionais
    const selectRegion = document.getElementById('select-region');
    selectRegion.innerHTML = '<option value="Todas">Todas as Regionais</option>';
    data.regional_offices.forEach(reg => {
      const opt = document.createElement('option');
      opt.value = reg;
      opt.textContent = reg;
      selectRegion.appendChild(opt);
    });

    // Gerentes
    const selectManager = document.getElementById('select-manager');
    selectManager.innerHTML = '<option value="Todos">Todos os Gerentes</option>';
    data.managers.forEach(mgr => {
      const opt = document.createElement('option');
      opt.value = mgr;
      opt.textContent = mgr;
      selectManager.appendChild(opt);
    });

    // Quadrantes dinâmicos
    const selectQuadrant = document.getElementById('select-quadrant');
    if (selectQuadrant && data.quadrants) {
      selectQuadrant.innerHTML = '<option value="Todos">Todos os Segmentos</option>';
      data.quadrants.forEach(q => {
        const opt = document.createElement('option');
        opt.value = q;
        opt.textContent = formatQuadrantShort(q);
        selectQuadrant.appendChild(opt);
      });
    }

    updateContextTag();
  } catch (err) {
    console.error('Falha ao carregar metadados do backend:', err);
  }
}

function updateContextTag() {
  const contextEl = document.getElementById('context-info-text');
  if (STATE.filter.mode === 'agent') {
    contextEl.textContent = `Vendedor: ${STATE.filter.agent} | Fila de Trabalho Operacional Ativa`;
  } else {
    contextEl.textContent = `Visão Executiva: Regional ${STATE.filter.region} | Gerência ${STATE.filter.manager}`;
  }
}

// ==============================================================================
// CARGA DE DADOS DO BACKEND
// ==============================================================================
async function refreshAllData() {
  await fetchDeals();
  await Promise.all([
    fetchZombies(),
    loadCharts()
  ]);
}

async function fetchDeals() {
  const params = new URLSearchParams();
  if (STATE.filter.mode === 'agent') {
    if (STATE.filter.agent) params.set('agent', STATE.filter.agent);
  } else {
    if (STATE.filter.region && STATE.filter.region !== 'Todas') params.set('region', STATE.filter.region);
    if (STATE.filter.manager && STATE.filter.manager !== 'Todos') params.set('manager', STATE.filter.manager);
  }

  if (STATE.filter.quadrant && STATE.filter.quadrant !== 'Todos') params.set('quadrant', STATE.filter.quadrant);
  if (STATE.filter.stage && STATE.filter.stage !== 'Todos') params.set('stage', STATE.filter.stage);
  if (STATE.filter.search) params.set('search', STATE.filter.search);

  params.set('sort_by', STATE.filter.sortBy);
  params.set('sort_asc', STATE.filter.sortAsc ? 'true' : 'false');
  params.set('limit', STATE.filter.limit);

  try {
    const res = await fetch(`/api/deals?${params.toString()}`);
    const data = await res.json();
    STATE.currentDeals = data.deals;

    // Atualiza KPIs globais do filtro
    updateKPIs(data.stats);

    // Renderiza Spotlight e Tabela
    renderSpotlight(data.deals);
    renderDealsTable(data.deals);

    document.getElementById('table-counter-text').textContent = 
      `Exibindo ${data.deals.filter(d => !d.matrix_quadrant.includes('Quarentena')).length} negócios vivos na fila (de ${data.total} no CRM)`;
  } catch (err) {
    console.error('Erro ao buscar oportunidades:', err);
  }
}

function updateKPIs(stats) {
  document.getElementById('val-total-deals').textContent = stats.total_deals.toLocaleString();
  document.getElementById('val-active-deals').textContent = stats.active_deals.toLocaleString();
  document.getElementById('val-zombie-delta').textContent = `-${stats.zombies} zumbis expurgados`;
  document.getElementById('val-pipeline-usd').textContent = `$${Math.round(stats.active_pipeline_val).toLocaleString()}`;
  document.getElementById('val-erv-usd').textContent = `$${stats.total_erv_per_day.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}/dia`;
  document.getElementById('val-quarantine-count').textContent = stats.zombies.toLocaleString();
}

// ==============================================================================
// RENDERIZAÇÃO: SPOTLIGHT DO LEAD #1
// ==============================================================================
function renderSpotlight(deals) {
  const spotlightCard = document.getElementById('hero-spotlight-card');
  const liveDeals = deals.filter(d => !d.matrix_quadrant.includes('Quarentena'));

  if (liveDeals.length === 0) {
    spotlightCard.style.display = 'none';
    return;
  }

  // O Lead #1 é o deal com maior lead_score
  const top = liveDeals[0];
  spotlightCard.style.display = 'block';

  document.getElementById('spot-opp-id').textContent = top.opportunity_id;
  document.getElementById('spot-product').textContent = top.product;
  document.getElementById('spot-price').textContent = `$${Math.round(top.sales_price).toLocaleString()}`;
  document.getElementById('spot-account').textContent = top.account_display;
  document.getElementById('spot-agent').textContent = top.sales_agent;
  document.getElementById('spot-age').textContent = `${top.deal_age_days} dias`;
  document.getElementById('spot-stage').textContent = top.deal_stage;

  document.getElementById('spot-score').textContent = `${top.lead_score.toFixed(1)}/100`;
  document.getElementById('spot-prob').textContent = `${(top.calibrated_prob_won * 100).toFixed(1)}%`;
  document.getElementById('spot-erv').textContent = `$${top.erv_usd_per_day.toFixed(2)}/dia`;
  document.getElementById('spot-playbook').textContent = top.action_playbook;

  const btnDiagnose = document.getElementById('btn-spot-diagnose');
  btnDiagnose.onclick = () => openDealDrawer(top.opportunity_id);
}

// ==============================================================================
// RENDERIZAÇÃO: TABELA DE NEGOCIAÇÕES VIVAS
// ==============================================================================
function renderDealsTable(deals) {
  const tbody = document.getElementById('deals-table-body');
  const liveDeals = deals.filter(d => !d.matrix_quadrant.includes('Quarentena'));

  if (liveDeals.length === 0) {
    tbody.innerHTML = `<tr><td colspan="12" class="table-empty">Nenhuma oportunidade ativa atende aos filtros atuais.</td></tr>`;
    return;
  }

  tbody.innerHTML = '';
  liveDeals.forEach(deal => {
    const tr = document.createElement('tr');
    tr.onclick = (e) => {
      if (!e.target.closest('button')) {
        openDealDrawer(deal.opportunity_id);
      }
    };

    const qClass = getQuadrantClass(deal.matrix_quadrant);

    tr.innerHTML = `
      <td class="cell-id"><strong>${deal.opportunity_id}</strong></td>
      <td class="cell-agent">${deal.sales_agent}</td>
      <td class="cell-product">${deal.product}</td>
      <td class="cell-account">${deal.account_display}</td>
      <td>${deal.deal_stage}</td>
      <td>${deal.deal_age_days}d</td>
      <td class="cell-price">$${Math.round(deal.sales_price).toLocaleString()}</td>
      <td class="cell-prob">${(deal.calibrated_prob_won * 100).toFixed(1)}%</td>
      <td class="cell-erv">$${deal.erv_usd_per_day.toFixed(2)}/d</td>
      <td class="cell-score">${deal.lead_score.toFixed(1)}</td>
      <td><span class="quadrant-badge ${qClass}">${formatQuadrantShort(deal.matrix_quadrant)}</span></td>
      <td>
        <button class="btn-tactical table-btn-action" onclick="window.openDealDrawer('${deal.opportunity_id}')">
          DIAGNÓSTICO
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function getQuadrantClass(quadrant) {
  if (!quadrant) return 'prospecting';
  if (quadrant.includes('Hot Closers') || quadrant.includes('Q1')) return 'q1';
  if (quadrant.includes('High-Value Stalling') || quadrant.includes('Q2')) return 'q2';
  if (quadrant.includes('Quick Wins') || quadrant.includes('Q3')) return 'q3';
  if (quadrant.includes('Quarentena') || quadrant.includes('Q4') || quadrant.includes('Zumbis')) return 'q4';
  if (quadrant.includes('Prospecção') || quadrant.includes('Ativação')) return 'prospecting';
  if (quadrant.includes('Cadência')) return 'cadence';
  return 'prospecting';
}

function formatQuadrantShort(quadrant) {
  if (!quadrant) return 'Prospecção Ativa';
  if (quadrant.includes('Hot Closers') || quadrant.includes('Q1')) return 'Q1: Hot Closer';
  if (quadrant.includes('High-Value Stalling') || quadrant.includes('Q2')) return 'Q2: Stalling';
  if (quadrant.includes('Quick Wins') || quadrant.includes('Q3')) return 'Q3: Quick Win';
  if (quadrant.includes('Quarentena') || quadrant.includes('Q4')) return 'Q4: Quarentena';
  if (quadrant.includes('Prospecção') || quadrant.includes('Ativação')) return 'Prospecção Ativa';
  if (quadrant.includes('Cadência')) return 'Cadência Normal';
  return quadrant;
}

// ==============================================================================
// TAB 2: CERCA SANITÁRIA (ZUMBIS)
// ==============================================================================
async function fetchZombies() {
  const params = new URLSearchParams();
  if (STATE.filter.mode === 'agent' && STATE.filter.agent) {
    params.set('agent', STATE.filter.agent);
  } else {
    if (STATE.filter.region && STATE.filter.region !== 'Todas') params.set('region', STATE.filter.region);
    if (STATE.filter.manager && STATE.filter.manager !== 'Todos') params.set('manager', STATE.filter.manager);
  }
  params.set('is_zombie', 'true');
  params.set('limit', '500');

  try {
    const res = await fetch(`/api/deals?${params.toString()}`);
    const data = await res.json();
    const zombies = data.deals;

    const totalCount = zombies.length;
    const totalVal = zombies.reduce((acc, curr) => acc + (curr.sales_price || 0), 0);
    const avgAge = totalCount > 0 ? (zombies.reduce((acc, curr) => acc + (curr.deal_age_days || 0), 0) / totalCount) : 0;

    document.getElementById('qz-count').textContent = totalCount.toLocaleString();
    document.getElementById('qz-volume').textContent = `$${Math.round(totalVal).toLocaleString()}`;
    document.getElementById('qz-avg-age').textContent = `${Math.round(avgAge)} dias`;

    document.getElementById('quarantine-counter-text').textContent = 
      `Exibindo ${totalCount} negócios retidos na cerca sanitária para arquivamento ou nutrição`;

    const tbody = document.getElementById('zombies-table-body');
    if (zombies.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="table-empty">Nenhum zumbi retido para o filtro selecionado.</td></tr>`;
      return;
    }

    tbody.innerHTML = '';
    zombies.forEach(z => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="cell-id"><strong>${z.opportunity_id}</strong></td>
        <td>${z.sales_agent}</td>
        <td>${z.product}</td>
        <td>${z.account_display}</td>
        <td>${z.deal_age_days} dias</td>
        <td>$${Math.round(z.sales_price).toLocaleString()}</td>
        <td style="color:var(--cyber-crimson); font-size:11.5px;">${z.risk_bottlenecks || 'Estagnação temporal crítica (>110 dias)'}</td>
        <td>${z.action_playbook || 'Arquivar ou transferir para automação'}</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error('Erro ao carregar zumbis da cerca sanitária:', err);
  }
}

// ==============================================================================
// TAB 3: DIAGNÓSTICO VISUAL & CHARTS
// ==============================================================================
async function loadCharts() {
  if (typeof Chart === 'undefined') return;

  const deals = STATE.currentDeals.filter(d => !d.matrix_quadrant.includes('Quarentena'));

  // Chart 1: Scatter (Deal Age vs ERV)
  const scatterCanvas = document.getElementById('scatterChart');
  if (scatterCanvas) {
    const ctx = scatterCanvas.getContext('2d');
    if (STATE.charts.scatter) STATE.charts.scatter.destroy();

    const dataQ1 = deals.filter(d => d.matrix_quadrant.includes('Q1')).map(d => ({ x: d.deal_age_days, y: d.erv_usd_per_day, r: Math.max(4, Math.min(18, d.sales_price / 1000 * 2)), deal: d }));
    const dataQ2 = deals.filter(d => d.matrix_quadrant.includes('Q2')).map(d => ({ x: d.deal_age_days, y: d.erv_usd_per_day, r: Math.max(4, Math.min(18, d.sales_price / 1000 * 2)), deal: d }));
    const dataQ3 = deals.filter(d => d.matrix_quadrant.includes('Q3')).map(d => ({ x: d.deal_age_days, y: d.erv_usd_per_day, r: Math.max(4, Math.min(18, d.sales_price / 1000 * 2)), deal: d }));
    const dataProsp = deals.filter(d => d.matrix_quadrant.includes('Prospecção') || d.matrix_quadrant.includes('Ativação')).map(d => ({ x: d.deal_age_days, y: d.erv_usd_per_day, r: Math.max(4, Math.min(18, d.sales_price / 1000 * 2)), deal: d }));
    const dataCadence = deals.filter(d => d.matrix_quadrant.includes('Cadência')).map(d => ({ x: d.deal_age_days, y: d.erv_usd_per_day, r: Math.max(4, Math.min(18, d.sales_price / 1000 * 2)), deal: d }));

    STATE.charts.scatter = new Chart(ctx, {
      type: 'bubble',
      data: {
        datasets: [
          { label: 'Q1: Hot Closers', data: dataQ1, backgroundColor: 'rgba(0, 255, 157, 0.75)', borderColor: '#00FF9D', borderWidth: 1 },
          { label: 'Q2: High-Value Stalling', data: dataQ2, backgroundColor: 'rgba(255, 184, 0, 0.75)', borderColor: '#FFB800', borderWidth: 1 },
          { label: 'Q3: Quick Wins', data: dataQ3, backgroundColor: 'rgba(0, 240, 255, 0.75)', borderColor: '#00F0FF', borderWidth: 1 },
          { label: 'Prospecção Ativa', data: dataProsp, backgroundColor: 'rgba(56, 189, 248, 0.65)', borderColor: '#38bdf8', borderWidth: 1 },
          { label: 'Cadência Normal', data: dataCadence, backgroundColor: 'rgba(192, 132, 252, 0.65)', borderColor: '#c084fc', borderWidth: 1 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#F5F5F5', font: { family: 'JetBrains Mono', size: 10 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const d = ctx.raw.deal;
                return `${d.opportunity_id} (${d.product}): $${d.sales_price.toLocaleString()} | Idade: ${d.deal_age_days}d | ERV: $${d.erv_usd_per_day.toFixed(2)}/dia`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Dias em Negociação', color: '#8B8DA8', font: { family: 'JetBrains Mono' } },
            ticks: { color: '#8B8DA8' },
            grid: { color: 'rgba(97, 97, 145, 0.15)' }
          },
          y: {
            title: { display: true, text: 'Throughput ERV ($/dia)', color: '#8B8DA8', font: { family: 'JetBrains Mono' } },
            ticks: { color: '#8B8DA8' },
            grid: { color: 'rgba(97, 97, 145, 0.15)' }
          }
        }
      }
    });
  }

  // Chart 2: Volume Quadrantes
  const quadrantsCanvas = document.getElementById('quadrantsChart');
  if (quadrantsCanvas) {
    const ctx = quadrantsCanvas.getContext('2d');
    if (STATE.charts.quadrants) STATE.charts.quadrants.destroy();

    const counts = {
      'Q1: Hot Closers': STATE.currentDeals.filter(d => d.matrix_quadrant.includes('Q1')).length,
      'Q2: Stalling': STATE.currentDeals.filter(d => d.matrix_quadrant.includes('Q2')).length,
      'Q3: Quick Wins': STATE.currentDeals.filter(d => d.matrix_quadrant.includes('Q3')).length,
      'Prospecção': STATE.currentDeals.filter(d => d.matrix_quadrant.includes('Prospecção') || d.matrix_quadrant.includes('Ativação')).length,
      'Cadência Normal': STATE.currentDeals.filter(d => d.matrix_quadrant.includes('Cadência')).length,
      'Q4: Quarentena': STATE.currentDeals.filter(d => d.matrix_quadrant.includes('Quarentena')).length
    };

    STATE.charts.quadrants = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: Object.keys(counts),
        datasets: [{
          data: Object.values(counts),
          backgroundColor: [
            'rgba(0, 255, 157, 0.85)',
            'rgba(255, 184, 0, 0.85)',
            'rgba(0, 240, 255, 0.85)',
            'rgba(56, 189, 248, 0.85)',
            'rgba(192, 132, 252, 0.85)',
            'rgba(255, 51, 102, 0.85)'
          ],
          borderColor: '#07091B',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#F5F5F5', font: { family: 'JetBrains Mono', size: 10 } } }
        }
      }
    });
  }
}

// ==============================================================================
// 3D CYBER DRAGON MASCOT ENGINE (THREE.JS PROCEDURAL)
// ==============================================================================
function initDraco3DMascots() {
  if (typeof THREE === 'undefined') {
    console.warn('Three.js não carregado. Operando com render fallback.');
    return;
  }

  // 1. Floating Mascot Canvas
  const mascotCanvas = document.getElementById('draco-mascot-canvas');
  if (mascotCanvas) {
    STATE.three3D.mascot = createCyberDragonScene(mascotCanvas, {
      size: 90,
      cameraZ: 4.2,
      orbitSpeed: 0.015,
      interactiveMouse: true
    });
  }

  // 2. Drawer Mascot Canvas
  const drawerCanvas = document.getElementById('draco-drawer-canvas');
  if (drawerCanvas) {
    STATE.three3D.drawer = createCyberDragonScene(drawerCanvas, {
      size: 100,
      cameraZ: 4.0,
      orbitSpeed: 0.02,
      interactiveMouse: false
    });
  }
}

function createCyberDragonScene(canvas, opts) {
  const width = opts.size || 100;
  const height = opts.size || 100;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.z = opts.cameraZ || 4.2;

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    alpha: true,
    antialias: true
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Iluminação Sci-Fi
  const ambLight = new THREE.AmbientLight(0x0a1236, 1.5);
  scene.add(ambLight);

  const dirLight = new THREE.DirectionalLight(0x00F0FF, 2.5);
  dirLight.position.set(2, 4, 3);
  scene.add(dirLight);

  const cobaltLight = new THREE.PointLight(0x0000F2, 4, 8);
  cobaltLight.position.set(0, 0, 1);
  scene.add(cobaltLight);

  const greenRimLight = new THREE.PointLight(0x00FF9D, 2, 5);
  greenRimLight.position.set(-2, -1, 1);
  scene.add(greenRimLight);

  // GRUPO PRINCIPAL DO DRAGÃO CIBERNÉTICO
  const dragon = new THREE.Group();

  // Materiais Cyberpunk
  const darkMetal = new THREE.MeshStandardMaterial({
    color: 0x070b24,
    roughness: 0.25,
    metalness: 0.9
  });

  const neonCyan = new THREE.MeshBasicMaterial({
    color: 0x00F0FF,
    wireframe: true
  });

  const neonCobalt = new THREE.MeshStandardMaterial({
    color: 0x0000F2,
    emissive: 0x0000B0,
    roughness: 0.2,
    metalness: 0.8
  });

  const eyeMaterial = new THREE.MeshBasicMaterial({
    color: 0x00FF9D
  });

  // 1. Crânio / Cabeça Facetada
  const headGeo = new THREE.ConeGeometry(0.55, 1.1, 5);
  headGeo.rotateX(Math.PI * 0.45);
  const headMesh = new THREE.Mesh(headGeo, darkMetal);
  dragon.add(headMesh);

  // Wireframe Glow na Cabeça
  const headWire = new THREE.Mesh(headGeo, neonCyan);
  headWire.scale.set(1.02, 1.02, 1.02);
  dragon.add(headWire);

  // 2. Chifres / Antenas Táticas
  const hornGeo = new THREE.ConeGeometry(0.09, 0.8, 4);
  const leftHorn = new THREE.Mesh(hornGeo, neonCobalt);
  leftHorn.position.set(-0.25, 0.4, -0.3);
  leftHorn.rotation.set(-0.5, 0, -0.4);
  dragon.add(leftHorn);

  const rightHorn = new THREE.Mesh(hornGeo, neonCobalt);
  rightHorn.position.set(0.25, 0.4, -0.3);
  rightHorn.rotation.set(-0.5, 0, 0.4);
  dragon.add(rightHorn);

  // 3. Olhos Laser Cyberpunk
  const eyeGeo = new THREE.SphereGeometry(0.065, 8, 8);
  const leftEye = new THREE.Mesh(eyeGeo, eyeMaterial);
  leftEye.position.set(-0.2, 0.12, 0.4);
  dragon.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeo, eyeMaterial);
  rightEye.position.set(0.2, 0.12, 0.4);
  dragon.add(rightEye);

  // 4. Asas Cibernéticas Articuladas (Delta Wings)
  const wingShape = new THREE.BufferGeometry();
  const wingVertices = new Float32Array([
    0.0, 0.0, 0.0,
    1.2, 0.6, -0.2,
    0.8, -0.5, -0.1,

    0.0, 0.0, 0.0,
    -1.2, 0.6, -0.2,
    -0.8, -0.5, -0.1
  ]);
  wingShape.setAttribute('position', new THREE.BufferAttribute(wingVertices, 3));
  wingShape.computeVertexNormals();

  const wingsMesh = new THREE.Mesh(wingShape, new THREE.MeshStandardMaterial({
    color: 0x000091,
    emissive: 0x000030,
    side: THREE.DoubleSide,
    metalness: 0.8
  }));
  wingsMesh.position.set(0, -0.1, -0.2);
  dragon.add(wingsMesh);

  // 5. Núcleo Holográfico Interno (Energy Core)
  const coreGeo = new THREE.OctahedronGeometry(0.28, 0);
  const coreMesh = new THREE.Mesh(coreGeo, new THREE.MeshBasicMaterial({
    color: 0x00F0FF,
    wireframe: true
  }));
  coreMesh.position.set(0, -0.15, 0.1);
  dragon.add(coreMesh);

  // 6. Anel Orbital de Telemetria
  const ringGeo = new THREE.TorusGeometry(1.4, 0.015, 6, 40);
  const ringMesh = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
    color: 0x00F0FF,
    transparent: true,
    opacity: 0.65
  }));
  ringMesh.rotation.x = Math.PI * 0.35;
  dragon.add(ringMesh);

  // 7. Partículas Orbitantes
  const particleCount = 28;
  const particleGeo = new THREE.BufferGeometry();
  const particlePos = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const r = 1.3 + (Math.random() - 0.5) * 0.4;
    particlePos[i * 3] = Math.cos(angle) * r;
    particlePos[i * 3 + 1] = (Math.random() - 0.5) * 0.6;
    particlePos[i * 3 + 2] = Math.sin(angle) * r;
  }
  particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
  const particleMat = new THREE.PointsMaterial({
    color: 0x00FF9D,
    size: 0.06,
    transparent: true,
    opacity: 0.8
  });
  const particles = new THREE.Points(particleGeo, particleMat);
  dragon.add(particles);

  scene.add(dragon);

  let mouseX = 0;
  let mouseY = 0;

  if (opts.interactiveMouse) {
    window.addEventListener('mousemove', (e) => {
      const halfW = window.innerWidth / 2;
      const halfH = window.innerHeight / 2;
      mouseX = (e.clientX - halfW) / halfW;
      mouseY = (e.clientY - halfH) / halfH;
    });
  }

  let clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Flutuação suave (breathing)
    dragon.position.y = Math.sin(t * 1.8) * 0.09;

    // Rotação autônoma sutil e acompanhamento do mouse
    dragon.rotation.y = Math.sin(t * 0.5) * 0.2 + mouseX * 0.4;
    dragon.rotation.x = mouseY * -0.25;

    // Pulsação e rotação do anel orbital
    ringMesh.rotation.z += 0.008;
    ringMesh.rotation.x += 0.004;

    // Rotação do núcleo interno
    coreMesh.rotation.y += 0.02;
    coreMesh.rotation.x += 0.015;

    // Órbita de partículas
    particles.rotation.y += opts.orbitSpeed || 0.012;

    renderer.render(scene, camera);
  }

  animate();

  return { scene, camera, renderer, dragon };
}

// ==============================================================================
// HUD DRAWER & COPILOTO LLM (OPENROUTER & DRACO CHAT)
// ==============================================================================
window.openDealDrawer = async function(oppId) {
  const drawer = document.getElementById('deal-drawer');
  drawer.classList.add('open');

  const titleEl = document.getElementById('drawer-opp-id');
  titleEl.textContent = oppId ? `DIAGNÓSTICO: ${oppId}` : 'DRACO AI // CONSOLE';

  const telemetrySection = document.getElementById('drawer-telemetry-section');
  const btnRunCopilot = document.getElementById('btn-run-copilot');

  if (oppId) {
    telemetrySection.style.display = 'block';
    btnRunCopilot.style.display = 'inline-flex';
    document.getElementById('copilot-output').style.display = 'none';
    document.getElementById('copilot-run-status').textContent = 'Pronto';
    btnRunCopilot.disabled = false;

    try {
      const res = await fetch(`/api/deal/${oppId}`);
      const deal = await res.json();
      STATE.selectedDeal = deal;

      // Preenche telemetria
      document.getElementById('d-agent').textContent = deal.sales_agent || '-';
      document.getElementById('d-product').textContent = deal.product || '-';
      document.getElementById('d-price').textContent = `$${Math.round(deal.sales_price).toLocaleString()}`;
      document.getElementById('d-account').textContent = deal.account_display || 'Não Informada';
      document.getElementById('d-stage').textContent = deal.deal_stage || '-';
      document.getElementById('d-age').textContent = `${deal.deal_age_days} dias`;
      document.getElementById('d-prob').textContent = `${(deal.calibrated_prob_won * 100).toFixed(1)}%`;
      document.getElementById('d-erv').textContent = `$${deal.erv_usd_per_day.toFixed(2)}/dia`;
      document.getElementById('d-score').textContent = `${deal.lead_score.toFixed(1)}/100`;
      document.getElementById('d-quadrant').textContent = formatQuadrantShort(deal.matrix_quadrant);

      // Decomposição matemática
      document.getElementById('d-prob-base').textContent = `${(deal.prob_base * 100).toFixed(1)}%`;
      document.getElementById('d-survival-factor').textContent = `${deal.survival_factor.toFixed(2)}x`;
      document.getElementById('d-quarter-modulator').textContent = `${deal.quarter_modulator.toFixed(2)}x`;
      document.getElementById('d-account-multiplier').textContent = `${deal.account_multiplier.toFixed(2)}x`;
      document.getElementById('d-risks').textContent = deal.risk_bottlenecks || 'Nenhum gargalo identificado';
      document.getElementById('d-drivers').textContent = deal.positive_drivers || 'Padrão';

      btnRunCopilot.onclick = () => runCopilotAnalysis(oppId);

    } catch (err) {
      console.error('Erro ao abrir detalhes da oportunidade:', err);
    }
  } else {
    // Abertura genérica pelo mascote
    document.getElementById('input-draco-chat').focus();
  }
};

function closeDealDrawer() {
  document.getElementById('deal-drawer').classList.remove('open');
}

async function runCopilotAnalysis(oppId) {
  const btn = document.getElementById('btn-run-copilot');
  const statusEl = document.getElementById('copilot-run-status');
  const outputEl = document.getElementById('copilot-output');

  btn.disabled = true;
  statusEl.textContent = 'Processando com IA...';
  statusEl.style.color = 'var(--cyber-cyan)';

  const payload = {
    opportunity_id: oppId,
    provider: STATE.settings.provider,
    openrouter_key: STATE.settings.openrouterKey || null,
    openai_key: STATE.settings.openaiKey || null,
    model: STATE.settings.provider === 'openrouter' ? STATE.settings.openrouterModel : null,
    temperature: STATE.settings.temperature
  };

  try {
    const res = await fetch('/api/copilot/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    const a = data.analysis;

    document.getElementById('co-resumo').textContent = a.resumo_executivo;
    document.getElementById('co-risco').textContent = a.diagnostico_risco;
    document.getElementById('co-alavanca').textContent = a.alavanca_comercial;
    document.getElementById('co-acao').textContent = a.acao_imediata;
    document.getElementById('co-script').textContent = a.roteiro_contato;

    outputEl.style.display = 'flex';
    if (data.is_fallback) {
      statusEl.textContent = 'Concluído (Offline Determinístico Grounded)';
      statusEl.style.color = 'var(--cyber-cyan)';
    } else {
      statusEl.textContent = `Concluído (${data.model_used})`;
      statusEl.style.color = 'var(--cyber-green)';
    }

    // Botão de copiar script
    document.getElementById('btn-copy-script').onclick = () => {
      navigator.clipboard.writeText(a.roteiro_contato);
      const b = document.getElementById('btn-copy-script');
      b.textContent = 'COPIADO!';
      setTimeout(() => { b.textContent = 'COPIAR SCRIPT'; }, 2000);
    };

  } catch (err) {
    console.error('Falha na inferência do copiloto:', err);
    statusEl.textContent = 'Erro na análise';
    statusEl.style.color = 'var(--cyber-crimson)';
  } finally {
    btn.disabled = false;
  }
}

// ==============================================================================
// CHAT CONVERSACIONAL DIRETO COM DRACO AI
// ==============================================================================
async function sendDracoChatMessage(userText) {
  const input = document.getElementById('input-draco-chat');
  const text = (userText || input.value).trim();
  if (!text) return;

  const messagesBox = document.getElementById('draco-chat-messages');
  const statusEl = document.getElementById('draco-chat-status');

  // Adiciona mensagem do usuário
  const userMsgEl = document.createElement('div');
  userMsgEl.className = 'chat-message user-msg';
  userMsgEl.innerHTML = `
    <div class="chat-sender">VOCÊ:</div>
    <div class="chat-bubble">${text}</div>
  `;
  messagesBox.appendChild(userMsgEl);

  input.value = '';
  messagesBox.scrollTop = messagesBox.scrollHeight;

  statusEl.textContent = 'Draco processando...';
  statusEl.style.color = 'var(--cyber-cyan)';

  // Contexto do CRM atual
  let contextStr = `Vendedor Ativo: ${STATE.filter.agent || 'Todos'}. Total Deals: ${STATE.currentDeals.length}.`;
  if (STATE.selectedDeal) {
    contextStr += ` Deal Atual: ${STATE.selectedDeal.opportunity_id} (${STATE.selectedDeal.product}), Ticket: $${STATE.selectedDeal.sales_price}, Idade: ${STATE.selectedDeal.deal_age_days} dias, Quadrante: ${STATE.selectedDeal.matrix_quadrant}.`;
  }

  const payload = {
    message: text,
    context_summary: contextStr,
    openrouter_key: STATE.settings.openrouterKey || null,
    openai_key: STATE.settings.openaiKey || null,
    provider: STATE.settings.provider,
    model: STATE.settings.provider === 'openrouter' ? STATE.settings.openrouterModel : null,
    temperature: 0.3
  };

  try {
    const res = await fetch('/api/copilot/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    const dracoMsgEl = document.createElement('div');
    dracoMsgEl.className = 'chat-message draco-msg';
    dracoMsgEl.innerHTML = `
      <div class="chat-sender">DRACO AI (${data.source || 'Grounded'}):</div>
      <div class="chat-bubble">${data.answer}</div>
    `;
    messagesBox.appendChild(dracoMsgEl);
    messagesBox.scrollTop = messagesBox.scrollHeight;

    statusEl.textContent = 'Pronto';
    statusEl.style.color = 'var(--cyber-green)';
  } catch (e) {
    console.error('Falha no chat do Draco:', e);
    statusEl.textContent = 'Erro no chat';
    statusEl.style.color = 'var(--cyber-crimson)';
  }
}

// ==============================================================================
// MODAL DE CONFIGURAÇÃO
// ==============================================================================
function openSettingsModal() {
  populateSettingsModal();
  document.getElementById('modal-settings').classList.add('open');
}

function closeSettingsModal() {
  document.getElementById('modal-settings').classList.remove('open');
}

// ==============================================================================
// EVENT LISTENERS GLOBAIS
// ==============================================================================
function setupEventListeners() {
  // Tabs Navigation
  document.querySelectorAll('.tab-button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
      STATE.activeTab = tabId;
      if (history.replaceState) {
        history.replaceState(null, null, '#' + tabId);
      }

      if (tabId === 'tab-visual') {
        setTimeout(loadCharts, 50);
      }
    });
  });

  // Suporte à navegação via Hash na URL (#tab-visual, #tab-quarantine, etc.)
  if (window.location.hash) {
    const hashTab = window.location.hash.replace('#', '');
    const targetBtn = document.querySelector(`.tab-button[data-tab="${hashTab}"]`);
    if (targetBtn) {
      setTimeout(() => targetBtn.click(), 100);
    }
  }

  // Suporte a parâmetros na URL (?deal=OPP_ID ou ?draco=1)
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('deal')) {
    setTimeout(() => window.openDealDrawer(urlParams.get('deal')), 300);
  } else if (urlParams.get('draco')) {
    setTimeout(() => window.openDealDrawer(null), 300);
  }

  // Profile Mode Toggle
  document.querySelectorAll('.btn-profile').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-profile').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const mode = btn.getAttribute('data-mode');
      STATE.filter.mode = mode;

      const wrapAgent = document.getElementById('wrap-agent-select');
      const wrapRegion = document.getElementById('wrap-region-select');
      const wrapManager = document.getElementById('wrap-manager-select');

      if (mode === 'agent') {
        wrapAgent.style.display = 'block';
        wrapRegion.style.display = 'none';
        wrapManager.style.display = 'none';
      } else {
        wrapAgent.style.display = 'none';
        wrapRegion.style.display = 'block';
        wrapManager.style.display = 'block';
      }

      updateContextTag();
      refreshAllData();
    });
  });

  // Selectors
  document.getElementById('select-agent').addEventListener('change', (e) => {
    STATE.filter.agent = e.target.value;
    updateContextTag();
    refreshAllData();
  });

  document.getElementById('select-region').addEventListener('change', (e) => {
    STATE.filter.region = e.target.value;
    updateContextTag();
    refreshAllData();
  });

  document.getElementById('select-manager').addEventListener('change', (e) => {
    STATE.filter.manager = e.target.value;
    updateContextTag();
    refreshAllData();
  });

  document.getElementById('select-quadrant').addEventListener('change', (e) => {
    STATE.filter.quadrant = e.target.value;
    refreshAllData();
  });

  document.getElementById('select-stage').addEventListener('change', (e) => {
    STATE.filter.stage = e.target.value;
    refreshAllData();
  });

  // Search input com debounce
  let searchTimeout = null;
  const inputSearch = document.getElementById('input-search');
  const btnClear = document.getElementById('btn-clear-search');

  inputSearch.addEventListener('input', (e) => {
    const val = e.target.value;
    btnClear.style.display = val ? 'block' : 'none';
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      STATE.filter.search = val;
      refreshAllData();
    }, 250);
  });

  btnClear.addEventListener('click', () => {
    inputSearch.value = '';
    btnClear.style.display = 'none';
    STATE.filter.search = '';
    refreshAllData();
  });

  // Tabela Sorting Headers
  document.querySelectorAll('#table-deals th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-sort');
      if (STATE.filter.sortBy === col) {
        STATE.filter.sortAsc = !STATE.filter.sortAsc;
      } else {
        STATE.filter.sortBy = col;
        STATE.filter.sortAsc = false;
      }
      document.querySelectorAll('#table-deals th').forEach(h => h.classList.remove('sort-active'));
      th.classList.add('sort-active');
      fetchDeals();
    });
  });

  // Exports
  document.getElementById('btn-export-active').addEventListener('click', () => {
    window.location.href = '/api/export/active';
  });

  document.getElementById('btn-export-zombies').addEventListener('click', () => {
    window.location.href = '/api/export/zombies';
  });

  // Settings Modal Controls
  document.getElementById('btn-open-settings').addEventListener('click', openSettingsModal);
  document.getElementById('btn-close-settings').addEventListener('click', closeSettingsModal);
  document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
  document.getElementById('btn-reset-settings').addEventListener('click', () => {
    localStorage.removeItem('nexus_revops_settings');
    localStorage.removeItem('nous_revops_settings');
    STATE.settings = {
      provider: 'openrouter',
      openrouterKey: '',
      openrouterModel: 'nousresearch/hermes-3-llama-3.1-70b',
      openaiKey: '',
      temperature: 0.2
    };
    populateSettingsModal();
    updateCopilotBadge();
  });

  document.getElementById('cfg-provider').addEventListener('change', (e) => {
    handleProviderChange(e.target.value);
  });

  document.getElementById('cfg-openrouter-model').addEventListener('change', (e) => {
    const custom = document.getElementById('cfg-custom-model');
    custom.style.display = e.target.value === 'custom' ? 'block' : 'none';
  });

  document.getElementById('cfg-temp').addEventListener('input', (e) => {
    document.getElementById('val-temp').textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Refresh Button
  document.getElementById('btn-refresh-data').addEventListener('click', () => {
    refreshAllData();
  });

  // Drawer Close
  document.getElementById('btn-close-drawer').addEventListener('click', closeDealDrawer);

  // CLIQUE NO MASCOTE 3D FLUTUANTE
  const floatingMascot = document.getElementById('draco-floating-mascot');
  if (floatingMascot) {
    floatingMascot.addEventListener('click', () => {
      window.openDealDrawer(null);
    });
  }

  // CHAT DIRETO COM O DRACO
  const btnSendChat = document.getElementById('btn-send-draco-chat');
  const inputChat = document.getElementById('input-draco-chat');
  if (btnSendChat && inputChat) {
    btnSendChat.addEventListener('click', () => sendDracoChatMessage());
    inputChat.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendDracoChatMessage();
    });
  }

  // Quick Action Chips do Draco
  document.querySelectorAll('.btn-quick-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      sendDracoChatMessage(prompt);
    });
  });
}
