function interpolateChannel(a, b, t) {
  return Math.round(a + (b - a) * t);
}

function interpolateColor(start, end, t) {
  return `rgb(${interpolateChannel(start[0], end[0], t)}, ${interpolateChannel(start[1], end[1], t)}, ${interpolateChannel(start[2], end[2], t)})`;
}

function onsiteEnergyColor(epsilon) {
  const maxAbs = Math.max(1, ...state.sites.map((site) => Math.abs(Number(site.epsilon) || 0)));
  const normalized = Math.max(-1, Math.min(1, (Number(epsilon) || 0) / maxAbs));
  if (normalized < 0) {
    return interpolateColor([37, 99, 235], [255, 255, 255], normalized + 1);
  }
  return interpolateColor([255, 255, 255], [220, 38, 38], normalized);
}

function modelBounds() {
  if (!state.sites.length) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  }
  const points = state.sites.map((site) => ({ x: site.x, y: site.y }));
  if (state.dimension === 2 && state.cell) {
    const origin = state.cell.origin || { x: 0, y: 0 };
    const a = state.cell.a;
    const b = state.cell.b;
    points.push(
      { x: origin.x, y: origin.y },
      { x: origin.x + a.x, y: origin.y + a.y },
      { x: origin.x + a.x + b.x, y: origin.y + a.y + b.y },
      { x: origin.x + b.x, y: origin.y + b.y },
    );
  }
  if (state.boundary === 'periodic') {
    state.bonds.filter((bond) => bond.periodic).forEach((bond) => {
      const source = siteById(bond.source);
      const target = siteById(bond.target);
      const offset = bond.offset || { x: 0, y: 0 };
      if (source) {
        points.push({ x: source.x - offset.x, y: source.y - offset.y });
      }
      if (target) {
        points.push({ x: target.x + offset.x, y: target.y + offset.y });
      }
    });
  }
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  return { minX, maxX, minY, maxY };
}

function modelToScreen(site) {
  const bounds = modelBounds();
  const padding = 90;
  const width = 1000 - 2 * padding;
  const height = 680 - 2 * padding;
  const baseScale = Math.min(
    width / Math.max(bounds.maxX - bounds.minX, 0.001),
    height / Math.max(bounds.maxY - bounds.minY, 0.001),
  );
  const scale = baseScale * state.view.zoom;
  const offsetX = 500 + state.view.panX - ((bounds.minX + bounds.maxX) / 2) * scale;
  const offsetY = 340 + state.view.panY + ((bounds.minY + bounds.maxY) / 2) * scale;
  return {
    x: offsetX + site.x * scale,
    y: offsetY - site.y * scale,
    scale,
  };
}

function screenToModel(clientX, clientY) {
  const point = canvas.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  const svgPoint = point.matrixTransform(canvas.getScreenCTM().inverse());
  const bounds = modelBounds();
  const padding = 90;
  const width = 1000 - 2 * padding;
  const height = 680 - 2 * padding;
  const baseScale = Math.min(
    width / Math.max(bounds.maxX - bounds.minX, 0.001),
    height / Math.max(bounds.maxY - bounds.minY, 0.001),
  );
  const scale = baseScale * state.view.zoom;
  const offsetX = 500 + state.view.panX - ((bounds.minX + bounds.maxX) / 2) * scale;
  const offsetY = 340 + state.view.panY + ((bounds.minY + bounds.maxY) / 2) * scale;
  return {
    x: (svgPoint.x - offsetX) / scale,
    y: -(svgPoint.y - offsetY) / scale,
  };
}

function renderAxes() {
  axisLayer.innerHTML = '';
  const origin = modelToScreen({ x: 0, y: 0 });
  const subtleClusterAxes = state.dimension === 2 && !isBlochRepresentation();
  const axisClass = subtleClusterAxes ? 'axis-line axis-line-subtle' : 'axis-line';
  const labelClass = subtleClusterAxes ? 'axis-label axis-label-subtle' : 'axis-label';
  const xAxis = createSvgElement('line', {
    x1: 40,
    y1: origin.y,
    x2: 960,
    y2: origin.y,
    class: axisClass,
  });
  const yAxis = createSvgElement('line', {
    x1: origin.x,
    y1: 40,
    x2: origin.x,
    y2: 640,
    class: axisClass,
  });
  const xLabel = createSvgElement('text', {
    x: 960,
    y: origin.y - 10,
    class: labelClass,
    'text-anchor': 'end',
  });
  xLabel.textContent = 'x';
  const yLabel = createSvgElement('text', {
    x: origin.x + 10,
    y: 52,
    class: labelClass,
  });
  yLabel.textContent = 'y';
  axisLayer.appendChild(xAxis);
  axisLayer.appendChild(yAxis);
  axisLayer.appendChild(xLabel);
  axisLayer.appendChild(yLabel);
}

function cellGridSegments(origin, a, b, subdivisions = {}) {
  const columns = Math.max(1, Math.round(Number(subdivisions.x) || 1));
  const rows = Math.max(1, Math.round(Number(subdivisions.y) || 1));
  const pointAt = (u, v) => ({
    x: origin.x + u * a.x + v * b.x,
    y: origin.y + u * a.y + v * b.y,
  });
  const segments = [];
  for (let column = 1; column < columns; column += 1) {
    const u = column / columns;
    segments.push({ start: pointAt(u, 0), end: pointAt(u, 1) });
  }
  for (let row = 1; row < rows; row += 1) {
    const v = row / rows;
    segments.push({ start: pointAt(0, v), end: pointAt(1, v) });
  }
  return segments;
}

function renderCellBoundary() {
  cellBoundaryLayer.innerHTML = '';
  if (state.dimension !== 2 || !state.cell) {
    return;
  }
  const origin = state.cell.origin || { x: 0, y: 0 };
  const a = state.cell.a;
  const b = state.cell.b;
  const corners = [
    origin,
    { x: origin.x + a.x, y: origin.y + a.y },
    { x: origin.x + a.x + b.x, y: origin.y + a.y + b.y },
    { x: origin.x + b.x, y: origin.y + b.y },
  ].map((point) => modelToScreen(point));
  cellGridSegments(origin, a, b, state.cellSubdivisions).forEach((segment) => {
    const start = modelToScreen(segment.start);
    const end = modelToScreen(segment.end);
    cellBoundaryLayer.appendChild(createSvgElement('line', {
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
      class: 'cell-grid-line',
    }));
  });
  const boundary = createSvgElement('polygon', {
    points: corners.map((point) => `${point.x},${point.y}`).join(' '),
    class: `cell-boundary ${state.boundary === 'periodic' ? 'periodic-cell' : 'open-cell'}`,
  });
  cellBoundaryLayer.appendChild(boundary);

  corners.forEach((point) => {
    cellBoundaryLayer.appendChild(createSvgElement('circle', {
      cx: point.x,
      cy: point.y,
      r: 3,
      class: 'cell-corner',
    }));
  });
}

function renderGhostLattice() {
  ghostLayer.innerHTML = '';
  const periodicBonds = state.bonds.filter((bond) => bond.periodic);
  if (!periodicBonds.length) {
    return;
  }
  periodicBonds.forEach((bond) => {
    const source = siteById(bond.source);
    const target = siteById(bond.target);
    if (!source || !target) {
      return;
    }
    const offset = bond.offset || { x: 0, y: 0 };
    const sourceMirror = modelToScreen({ x: source.x - offset.x, y: source.y - offset.y });
    const targetMirror = modelToScreen({ x: target.x + offset.x, y: target.y + offset.y });
    const sourcePoint = modelToScreen(source);
    const targetPoint = modelToScreen(target);
    [
      { start: sourcePoint, end: targetMirror },
      { start: sourceMirror, end: targetPoint },
    ].forEach((segment) => {
      ghostLayer.appendChild(createSvgElement('line', {
        x1: segment.start.x,
        y1: segment.start.y,
        x2: segment.end.x,
        y2: segment.end.y,
        class: 'ghost-bond',
      }));
    });
    [
      { point: sourceMirror, label: source.id },
      { point: targetMirror, label: target.id },
    ].forEach((ghost) => {
      ghostLayer.appendChild(createSvgElement('circle', {
        cx: ghost.point.x,
        cy: ghost.point.y,
        r: 13,
        class: 'ghost-site',
      }));
      const label = createSvgElement('text', {
        x: ghost.point.x,
        y: ghost.point.y + 5,
        class: 'ghost-site-label',
      });
      label.textContent = ghost.label;
      ghostLayer.appendChild(label);
    });
  });
}

function setZoom(nextZoom) {
  state.view.zoom = Math.max(0.35, Math.min(3, nextZoom));
  document.getElementById('zoom-label').textContent = `${Math.round(state.view.zoom * 100)}%`;
  render();
}

function setPanMode(enabled) {
  state.view.mode = enabled ? 'pan' : 'edit';
  document.getElementById('tool-pan').classList.toggle('active', enabled);
  canvas.classList.toggle('pan-mode', enabled);
}

function resetView() {
  state.view.zoom = 1;
  state.view.panX = 0;
  state.view.panY = 0;
  document.getElementById('zoom-label').textContent = '100%';
  render();
}

function selectItem(type, id) {
  state.selected = { type, id };
  state.pendingBondSource = null;
  render();
}

function setTool(tool) {
  state.tool = tool;
  state.pendingBondSource = null;
  document.querySelectorAll('.tool-button').forEach((button) => button.classList.remove('active'));
  document.getElementById(`tool-${tool === 'add_site' ? 'add-site' : tool === 'add_bond' ? 'add-bond' : 'select'}`).classList.add('active');
  updateSelectionSummary();
}

function createSvgElement(tag, attrs = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function renderBonds() {
  bondLayer.innerHTML = '';
  labelLayer.innerHTML = '';
  state.bonds.forEach((bond) => {
    const source = siteById(bond.source);
    const target = siteById(bond.target);
    if (!source || !target) {
      return;
    }
    const a = modelToScreen(source);
    const b = modelToScreen(target);
    const selected = state.selected && state.selected.type === 'bond' && state.selected.id === bond.id;
    const group = createSvgElement('g', { 'data-bond-id': bond.id });
    let labelX = (a.x + b.x) / 2;
    let labelY = (a.y + b.y) / 2 - 8;
    if (bond.periodic) {
      const vector = bondVector(bond);
      const length = Math.max(vector.length, 1e-8);
      const unit = { x: vector.dx / length, y: vector.dy / length };
      const halfLength = Math.min(38, Math.max(20, length * a.scale * 0.42));
      const sourceEnd = {
        x: a.x + unit.x * halfLength,
        y: a.y - unit.y * halfLength,
      };
      const targetEnd = {
        x: b.x - unit.x * halfLength,
        y: b.y + unit.y * halfLength,
      };
      [
        { start: a, end: sourceEnd },
        { start: b, end: targetEnd },
      ].forEach((segment) => {
        group.appendChild(createSvgElement('line', {
          x1: segment.start.x,
          y1: segment.start.y,
          x2: segment.end.x,
          y2: segment.end.y,
          class: 'bond-hit',
        }));
        group.appendChild(createSvgElement('line', {
          x1: segment.start.x,
          y1: segment.start.y,
          x2: segment.end.x,
          y2: segment.end.y,
          class: `bond periodic ${bond.kind || 'custom'}${selected ? ' selected' : ''}`,
        }));
      });
      labelX = (a.x + sourceEnd.x) / 2;
      labelY = (a.y + sourceEnd.y) / 2 - 8;
    } else {
      const hit = createSvgElement('line', {
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        class: 'bond-hit',
      });
      const line = createSvgElement('line', {
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        class: `bond ${bond.kind || 'custom'}${selected ? ' selected' : ''}`,
      });
      group.appendChild(hit);
      group.appendChild(line);
    }
    const labelParts = [];
    if (state.geometryCoupled) {
      labelParts.push(`t=${formatEnergyValue(effectiveBondT(bond))}`);
    }
    if (state.geometryCoupledV) {
      labelParts.push(`V=${formatEnergyValue(effectiveBondV(bond))}`);
    }
    if (labelParts.length) {
      const label = createSvgElement('text', {
        x: labelX,
        y: labelY,
        class: 'bond-label',
      });
      label.textContent = labelParts.join(' ');
      labelLayer.appendChild(label);
    }
    group.addEventListener('pointerdown', (event) => {
      event.stopPropagation();
      selectItem('bond', bond.id);
    });
    bondLayer.appendChild(group);
  });
}

function renderSites() {
  siteLayer.innerHTML = '';
  state.sites.forEach((site) => {
    const point = modelToScreen(site);
    const selected = state.selected && state.selected.type === 'site' && state.selected.id === site.id;
    const group = createSvgElement('g', {
      class: `site sublattice-${site.sublattice || 'a'}${selected ? ' selected' : ''}`,
      transform: `translate(${point.x}, ${point.y})`,
      'data-site-id': site.id,
    });
    const circle = createSvgElement('circle', { r: 24, class: 'site-circle' });
    circle.style.fill = onsiteEnergyColor(site.epsilon);
    group.appendChild(circle);
    const label = createSvgElement('text', { class: 'site-label' });
    label.textContent = site.id;
    group.appendChild(label);
    group.addEventListener('pointerdown', (event) => handleSitePointerDown(event, site.id));
    siteLayer.appendChild(group);
  });
}

function updateSelectionSummary() {
  const node = document.getElementById('selection-summary');
  if (state.tool === 'add_bond' && state.pendingBondSource !== null) {
    node.textContent = `Bond mode: source site ${state.pendingBondSource} selected`;
    return;
  }
  if (!state.selected) {
    node.textContent = state.tool === 'add_site' ? 'Click canvas to add a site' : 'No selection';
    return;
  }
  if (state.selected.type === 'site') {
    node.textContent = `Selected site ${state.selected.id}`;
  } else {
    const bond = bondById(state.selected.id);
    node.textContent = bond ? `Selected bond ${bond.source}-${bond.target}` : 'No selection';
  }
}

function updateInspector() {
  const empty = document.getElementById('empty-inspector');
  const sitePanel = document.getElementById('site-inspector');
  const bondPanel = document.getElementById('bond-inspector');
  const sitePhononFields = document.getElementById('site-phonon-fields');
  const blochBondFields = document.getElementById('bloch-bond-fields');
  empty.classList.remove('hidden');
  sitePanel.classList.add('hidden');
  bondPanel.classList.add('hidden');
  sitePhononFields.classList.add('hidden');
  blochBondFields.classList.add('hidden');

  if (!state.selected) {
    return;
  }
  if (state.selected.type === 'site') {
    const site = siteById(state.selected.id);
    if (!site) {
      return;
    }
    empty.classList.add('hidden');
    sitePanel.classList.remove('hidden');
    document.getElementById('site-id-label').textContent = site.id;
    setNumberValue('site-x', site.x);
    setNumberValue('site-y', site.y);
    setNumberValue('site-epsilon', site.epsilon);
    setNumberValue('site-u', site.U);
    sitePhononFields.classList.toggle('hidden', !isHolsteinHubbardModel());
    setNumberValue('site-phonon-omega', site.phononOmega);
    setNumberValue('site-electron-phonon-g', site.electronPhononG);
  } else {
    const bond = bondById(state.selected.id);
    if (!bond) {
      return;
    }
    empty.classList.add('hidden');
    bondPanel.classList.remove('hidden');
    document.getElementById('bond-id-label').textContent = `${bond.source}-${bond.target}`;
    document.getElementById('bond-source').value = bond.source;
    document.getElementById('bond-target').value = bond.target;
    setNumberValue('bond-t', bond.t);
    setNumberValue('bond-v', bond.V);
    blochBondFields.classList.toggle('hidden', !isBlochRepresentation());
    const cellOffset = bond.cellOffset || Array.from({ length: state.dimension }, () => 0);
    setNumberValue('bond-cell-offset-x', Number(cellOffset[0]) || 0);
    setNumberValue('bond-cell-offset-y', Number(cellOffset[1]) || 0);
    document.getElementById('bond-add-hc').checked = bond.addHermitianConjugate !== false;
  }
}

function updatePreview() {
  const preview = document.getElementById('preview');
  if (state.activeTab === 'summary') {
    preview.textContent = summaryText();
  } else if (state.activeTab === 'h1') {
    preview.textContent = isBlochRepresentation()
      ? `h(k = Gamma) (${energyUnitText()})\n${matrixPreview(h1Matrix())}`
      : `h1e (${energyUnitText()})\n${matrixPreview(h1Matrix())}`;
  } else if (state.activeTab === 'json') {
    preview.textContent = specJson();
  } else {
    preview.textContent = pythonInputText();
  }
}

function updateCanvasText() {
  const title = document.getElementById('canvas-title');
  const labels = {
    chain: '1D linear chain',
    ring: '1D ring',
    zigzag: '1D zigzag chain',
    quasi_1d: 'Quasi-1D ladder',
    square: '2D square',
    lieb: '2D Lieb',
    honeycomb: '2D honeycomb',
    triangular: '2D triangular',
    kagome: '2D kagome',
    dice: '2D dice',
    square_octagon: '2D square-octagon',
  };
  const representationLabel = isBlochRepresentation() ? ' primitive cell' : '';
  title.textContent = `${labels[state.preset] || state.preset}${representationLabel}`;
  const modulation = isSshBondModulationEnabled() ? ' | SSH alternation' : '';
  const modelLabel = isHolsteinHubbardModel() ? 'Holstein-Hubbard' : 'Hubbard';
  const modeLabel = isBlochRepresentation() ? 'Bloch' : 'Finite';
  document.getElementById('model-summary').textContent = `${modelLabel}${modulation} | ${modeLabel} | ${state.sites.length} sites | ${state.bonds.length} bonds`;
  updateSelectionSummary();
}

function render() {
  renderAxes();
  renderGhostLattice();
  renderCellBoundary();
  renderBonds();
  renderSites();
  updateInspector();
  updateCanvasText();
  updatePreview();
}
