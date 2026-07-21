const form = document.querySelector('#ask-form');
const questionInput = document.querySelector('#question');
const sendButton = document.querySelector('#send-button');
const loadingPanel = document.querySelector('#loading-panel');
const loadingTitle = document.querySelector('#loading-title');
const loadingCopy = document.querySelector('#loading-copy');
const loadingTime = document.querySelector('#loading-time');
const resultGrid = document.querySelector('#result-grid');
const errorPanel = document.querySelector('#error-panel');
const errorTitle = document.querySelector('#error-title');
const errorCopy = document.querySelector('#error-copy');
const answerText = document.querySelector('#answer-text');
const supportPill = document.querySelector('#support-pill');
const decisionLabel = document.querySelector('#decision-label');
const latencyLabel = document.querySelector('#latency-label');
const edgeCount = document.querySelector('#edge-count');
const graphElement = document.querySelector('#graph');
const graphEmpty = document.querySelector('#graph-empty');
const engineStatus = document.querySelector('#engine-status');

let network = null;
let loadingTimer = null;

document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => {
    questionInput.value = button.dataset.question;
    form.requestSubmit();
  });
});

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  setLoading(true);
  const started = performance.now();
  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload.detail || 'request_failed');
      error.status = response.status;
      error.retryAfter = response.headers.get('Retry-After');
      throw error;
    }
    renderResult(payload);
    setEngineStatus('ready', 'Demo online · bounded overlay');
  } catch (error) {
    renderError(error);
  } finally {
    setLoading(false, started);
  }
});

function setLoading(active) {
  clearInterval(loadingTimer);
  if (!active) {
    loadingPanel.hidden = true;
    sendButton.disabled = false;
    return;
  }

  resultGrid.hidden = true;
  errorPanel.hidden = true;
  loadingPanel.hidden = false;
  sendButton.disabled = true;
  const started = performance.now();
  loadingTitle.textContent = 'Reading the local graph…';
  loadingCopy.textContent = 'Selecting only evidence the planner can support.';
  loadingTime.textContent = '0.0s';

  loadingTimer = setInterval(() => {
    const seconds = (performance.now() - started) / 1000;
    loadingTime.textContent = `${seconds.toFixed(1)}s`;
    if (seconds > 20) {
      loadingTitle.textContent = 'Waking the demo instance…';
      loadingCopy.textContent = 'A free-tier cold start can take up to a minute. Your request is still running.';
    } else if (seconds > 6) {
      loadingTitle.textContent = 'Opening the bounded memory…';
      loadingCopy.textContent = 'The service may be waking after a period of inactivity.';
    }
  }, 100);
}

function renderResult(payload) {
  const used = Array.isArray(payload.edges_used) ? payload.edges_used : [];
  const context = Array.isArray(payload.context_edges) ? payload.context_edges : [];
  answerText.textContent = payload.answer || 'No answer text returned.';
  supportPill.textContent = humanize(payload.support_kind || 'unsupported');
  supportPill.classList.toggle('audit', payload.decision === 'audit');
  decisionLabel.textContent = `Decision: ${payload.decision || 'unknown'}`;
  latencyLabel.textContent = formatLatency(payload.latency_ms);
  edgeCount.textContent = `${used.length} ${used.length === 1 ? 'edge' : 'edges'} used`;
  resultGrid.hidden = false;
  errorPanel.hidden = true;
  renderGraph(used, context, payload.decision);
}

function renderError(error) {
  resultGrid.hidden = true;
  errorPanel.hidden = false;
  if (error.status === 429) {
    errorTitle.textContent = 'Rate limit reached.';
    errorCopy.textContent = `This public demo is protected from abuse. Try again in ${error.retryAfter || 'a few'} seconds.`;
    return;
  }
  if (error.status === 503) {
    errorTitle.textContent = 'The demo engine is briefly unavailable.';
    errorCopy.textContent = 'The instance may still be starting. Please try the same question again in a moment.';
    setEngineStatus('cold', 'Demo warming up');
    return;
  }
  errorTitle.textContent = 'The request could not be completed.';
  errorCopy.textContent = 'Check your connection and try again.';
}

function renderGraph(usedEdges, contextEdges, decision) {
  if (network) {
    network.destroy();
    network = null;
  }
  graphElement.replaceChildren();
  graphEmpty.hidden = usedEdges.length > 0;
  graphElement.hidden = usedEdges.length === 0;
  if (usedEdges.length === 0) {
    const emptyTitle = document.querySelector('#graph-empty-title');
    const emptyCopy = document.querySelector('#graph-empty-copy');
    if (decision === 'audit') {
      emptyTitle.textContent = 'No factual edge admitted';
      emptyCopy.textContent = 'The audit gate stopped before unsupported evidence could enter the answer.';
    } else {
      emptyTitle.textContent = 'Trace unavailable';
      emptyCopy.textContent = 'The answer was returned, but no explicit edge trace was provided.';
    }
    return;
  }

  const usedIds = new Set(usedEdges.map((edge) => edge.evidence_id));
  const edgeMap = new Map();
  [...contextEdges, ...usedEdges].forEach((edge) => edgeMap.set(edge.evidence_id, edge));
  const edges = [...edgeMap.values()];
  const activeNodes = new Set(usedEdges.flatMap((edge) => [edge.subject, edge.object]));
  const nodeNames = [...new Set(edges.flatMap((edge) => [edge.subject, edge.object]))];

  if (window.vis?.Network && window.vis?.DataSet) {
    const nodes = new window.vis.DataSet(nodeNames.map((name) => ({
      id: name,
      label: wrapLabel(name),
      shape: 'dot',
      size: activeNodes.has(name) ? 17 : 10,
      color: activeNodes.has(name)
        ? { background: '#111111', border: '#b9ff66', highlight: { background: '#171717', border: '#d6ffa4' } }
        : { background: '#161616', border: '#4b4b46', highlight: { background: '#202020', border: '#74746e' } },
      font: { color: activeNodes.has(name) ? '#f2f2ed' : '#777771', size: activeNodes.has(name) ? 13 : 10, face: 'SFMono-Regular, Consolas, monospace' },
      borderWidth: activeNodes.has(name) ? 2 : 1,
    })));
    const graphEdges = new window.vis.DataSet(edges.map((edge) => {
      const selected = usedIds.has(edge.evidence_id);
      return {
        id: edge.evidence_id,
        from: edge.subject,
        to: edge.object,
        label: edge.predicate.replaceAll('_', ' '),
        arrows: { to: { enabled: true, scaleFactor: .55 } },
        width: selected ? 2.6 : 1,
        color: { color: selected ? '#b9ff66' : '#3d3d39', highlight: selected ? '#d6ffa4' : '#5d5d57', opacity: selected ? 1 : .72 },
        font: { color: selected ? '#b9ff66' : '#5f5f59', size: selected ? 10 : 9, face: 'SFMono-Regular, Consolas, monospace', strokeWidth: 6, strokeColor: '#080808', align: 'top' },
        smooth: { enabled: true, type: 'dynamic' },
        dashes: !selected,
      };
    }));
    network = new window.vis.Network(graphElement, { nodes, edges: graphEdges }, {
      autoResize: true,
      interaction: { hover: true, tooltipDelay: 120, navigationButtons: false, keyboard: true },
      physics: {
        enabled: true,
        stabilization: { iterations: 180, fit: true },
        barnesHut: { gravitationalConstant: -3600, centralGravity: .18, springLength: 125, springConstant: .035, damping: .16, avoidOverlap: .45 },
      },
      layout: { improvedLayout: true },
      nodes: { shadow: false },
    });
    network.once('stabilizationIterationsDone', () => network.setOptions({ physics: { enabled: false } }));
    return;
  }

  renderSvgFallback(graphElement, nodeNames, edges, usedIds, activeNodes);
}

function renderSvgFallback(container, nodeNames, edges, usedIds, activeNodes) {
  const namespace = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(namespace, 'svg');
  svg.setAttribute('viewBox', '0 0 720 400');
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  const positions = new Map(nodeNames.map((name, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, nodeNames.length) - Math.PI / 2;
    const radius = nodeNames.length < 4 ? 115 : 155;
    return [name, { x: 360 + Math.cos(angle) * radius, y: 200 + Math.sin(angle) * radius }];
  }));
  edges.forEach((edge) => {
    const from = positions.get(edge.subject);
    const to = positions.get(edge.object);
    const line = document.createElementNS(namespace, 'line');
    line.setAttribute('x1', from.x); line.setAttribute('y1', from.y);
    line.setAttribute('x2', to.x); line.setAttribute('y2', to.y);
    line.setAttribute('stroke', usedIds.has(edge.evidence_id) ? '#b9ff66' : '#3d3d39');
    line.setAttribute('stroke-width', usedIds.has(edge.evidence_id) ? '3' : '1');
    if (!usedIds.has(edge.evidence_id)) line.setAttribute('stroke-dasharray', '5 5');
    svg.appendChild(line);
    const label = document.createElementNS(namespace, 'text');
    label.setAttribute('x', (from.x + to.x) / 2); label.setAttribute('y', (from.y + to.y) / 2 - 6);
    label.setAttribute('text-anchor', 'middle'); label.setAttribute('fill', usedIds.has(edge.evidence_id) ? '#b9ff66' : '#5f5f59');
    label.setAttribute('font-size', '10'); label.textContent = edge.predicate.replaceAll('_', ' ');
    svg.appendChild(label);
  });
  nodeNames.forEach((name) => {
    const point = positions.get(name);
    const circle = document.createElementNS(namespace, 'circle');
    circle.setAttribute('cx', point.x); circle.setAttribute('cy', point.y);
    circle.setAttribute('r', activeNodes.has(name) ? '14' : '9');
    circle.setAttribute('fill', activeNodes.has(name) ? '#b9ff66' : '#3d3d39');
    svg.appendChild(circle);
    const label = document.createElementNS(namespace, 'text');
    label.setAttribute('x', point.x); label.setAttribute('y', point.y + 30);
    label.setAttribute('text-anchor', 'middle'); label.setAttribute('fill', activeNodes.has(name) ? '#f2f2ed' : '#777771');
    label.setAttribute('font-size', activeNodes.has(name) ? '12' : '10'); label.textContent = name;
    svg.appendChild(label);
  });
  container.appendChild(svg);
}

function setEngineStatus(state, text) {
  engineStatus.className = `demo-status ${state}`;
  engineStatus.querySelector('[data-status-text]').textContent = text;
}

function wrapLabel(value) {
  const words = value.split(' ');
  if (value.length < 20 || words.length < 3) return value;
  const midpoint = Math.ceil(words.length / 2);
  return `${words.slice(0, midpoint).join(' ')}\n${words.slice(midpoint).join(' ')}`;
}

function humanize(value) {
  return String(value).replaceAll('_', ' ');
}

function formatLatency(value) {
  const number = Number(value || 0);
  if (number >= 1000) return `${(number / 1000).toFixed(2)} s`;
  return `${number.toFixed(1)} ms`;
}

fetch('/health', { cache: 'no-store' })
  .then((response) => response.json())
  .then((health) => {
    if (health.engine_status === 'ready') setEngineStatus('ready', 'Demo online · bounded overlay');
    else if (health.engine_status === 'cold') setEngineStatus('cold', 'Online · engine wakes on first ask');
    else setEngineStatus('cold', 'Demo warming up');
  })
  .catch(() => setEngineStatus('cold', 'Demo may be waking'));
