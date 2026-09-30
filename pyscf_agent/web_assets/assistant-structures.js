function resetModelHamiltonianStructurePreview(message) {
  lastModelHamiltonianPreview = null;
  const preview = document.getElementById('model-hamiltonian-structure-preview');
  if (preview) {
    preview.innerHTML = `<div class="model-structure-empty">${escapeHtml(message || t('modelHamiltonianPreviewEmpty'))}</div>`;
  }
  renderDmetSpinSector();
}

function dmetSpinSector(preview) {
  const validation = preview && preview.dmet_validation && typeof preview.dmet_validation === 'object'
    ? preview.dmet_validation
    : {};
  if (validation.spin_sector && typeof validation.spin_sector === 'object') {
    return validation.spin_sector;
  }
  const nelec = preview && Array.isArray(preview.nelec) ? preview.nelec : [];
  if (nelec.length !== 2 || !nelec.every((value) => Number.isInteger(Number(value)))) {
    return null;
  }
  const nalpha = Number(nelec[0]);
  const nbeta = Number(nelec[1]);
  return {
    nalpha,
    nbeta,
    nelec: nalpha + nbeta,
    libdmet_sz: nalpha - nbeta,
    physical_sz: 0.5 * (nalpha - nbeta),
    spin_multiplicity: Math.abs(nalpha - nbeta) + 1,
  };
}

function renderDmetSpinSector() {
  const target = document.getElementById('model-dmet-spin-sector-values');
  const validationTarget = document.getElementById('model-dmet-validation');
  const referenceControl = document.getElementById('model-dmet-reference');
  if (!target || !validationTarget || !referenceControl) return;
  referenceControl.classList.remove('input-invalid');
  referenceControl.removeAttribute('aria-invalid');
  const preview = lastModelHamiltonianPreview;
  const sector = dmetSpinSector(preview);
  const validation = preview && preview.dmet_validation && typeof preview.dmet_validation === 'object'
    ? preview.dmet_validation
    : null;
  if (!sector) {
    target.textContent = 'Load a model input to derive the spin sector.';
    validationTarget.textContent = '';
    return;
  }
  const requestedReference = referenceControl.value;
  const effectiveReference = validation && validation.effective_reference
    ? validation.effective_reference
    : requestedReference;
  const values = [
    ['N', sector.nelec],
    ['Nalpha', sector.nalpha],
    ['Nbeta', sector.nbeta],
    ['libDMET Sz', sector.libdmet_sz],
    ['physical Sz', sector.physical_sz],
    ['multiplicity', sector.spin_multiplicity || (Math.abs(sector.libdmet_sz) + 1)],
    ['reference', effectiveReference],
  ];
  if (validation && validation.execution_mode) {
    values.push(['mode', validation.execution_mode === 'translational' ? 'translated fragment' : 'finite graph']);
  }
  if (validation && validation.translation_backend === 'builder_primitive_cell') {
    values.push(['translation source', 'Builder primitive cell']);
    if (Array.isArray(validation.representative_site_ids)) {
      values.push(['representative sites', validation.representative_site_ids.join(', ')]);
    }
  }
  target.innerHTML = values.map(([label, value]) => (
    `<span class="dmet-spin-sector-value"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></span>`
  )).join('');
  if (!validation) {
    target.insertAdjacentHTML('beforeend', '<span class="dmet-spin-sector-status">Preview required for compatibility check</span>');
    validationTarget.textContent = '';
    return;
  }
  const errors = Array.isArray(validation.errors) ? validation.errors.filter(Boolean) : [];
  if (validation.valid && !errors.length) {
    target.insertAdjacentHTML('beforeend', '<span class="dmet-spin-sector-status valid">Compatible</span>');
    validationTarget.textContent = '';
    return;
  }
  validationTarget.textContent = errors.join(' | ') || 'The selected DMET settings are not compatible with this model.';
  if (sector.libdmet_sz !== 0 && requestedReference === 'restricted') {
    referenceControl.classList.add('input-invalid');
    referenceControl.setAttribute('aria-invalid', 'true');
  }
}

function invalidateDmetPreviewValidation() {
  if (lastModelHamiltonianPreview && typeof lastModelHamiltonianPreview === 'object') {
    delete lastModelHamiltonianPreview.dmet_validation;
  }
  renderDmetSpinSector();
}

function renderModelHamiltonianStructurePreview(preview) {
  const target = document.getElementById('model-hamiltonian-structure-preview');
  if (!target) {
    return;
  }
  if (!preview || typeof preview !== 'object') {
    target.innerHTML = `<div class="model-structure-empty">${escapeHtml(t('modelHamiltonianPreviewEmpty'))}</div>`;
    return;
  }
  const sites = Array.isArray(preview.sites_preview) ? preview.sites_preview : [];
  const bonds = Array.isArray(preview.bonds_preview) ? preview.bonds_preview : [];
  if (!sites.length) {
    target.innerHTML = `<div class="model-structure-empty">${escapeHtml(t('modelHamiltonianPreviewNoSites'))}</div>`;
    return;
  }
  const numericSites = sites.map((site) => ({
    id: site.id,
    x: Number.isFinite(Number(site.x)) ? Number(site.x) : 0,
    y: Number.isFinite(Number(site.y)) ? Number(site.y) : 0,
    epsilon: site.epsilon,
    U: site.U,
  }));
  const siteMap = new Map(numericSites.map((site) => [String(site.id), site]));
  const plotPoints = numericSites.map((site) => ({ x: site.x, y: site.y }));
  const cell = preview.cell && typeof preview.cell === 'object' ? preview.cell : null;
  if (preview.boundary === 'periodic' && cell && cell.a && cell.b) {
    const origin = cell.origin || { x: 0, y: 0 };
    const a = cell.a;
    const b = cell.b;
    plotPoints.push(
      { x: Number(origin.x) || 0, y: Number(origin.y) || 0 },
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) + (Number(b.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(b.y) || 0) },
    );
  }
  bonds.filter((bond) => bond.periodic).forEach((bond) => {
    const source = siteMap.get(String(bond.source));
    const targetSite = siteMap.get(String(bond.target));
    const offset = bond.offset || { x: 0, y: 0 };
    const dx = Number(offset.x) || 0;
    const dy = Number(offset.y) || 0;
    if (source) {
      plotPoints.push({ x: source.x - dx, y: source.y - dy });
    }
    if (targetSite) {
      plotPoints.push({ x: targetSite.x + dx, y: targetSite.y + dy });
    }
  });
  const minX = Math.min(...plotPoints.map((point) => point.x));
  const maxX = Math.max(...plotPoints.map((point) => point.x));
  const minY = Math.min(...plotPoints.map((point) => point.y));
  const maxY = Math.max(...plotPoints.map((point) => point.y));
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const width = 720;
  const height = 420;
  const padding = 46;
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const xOffset = (width - spanX * scale) / 2;
  const yOffset = (height - spanY * scale) / 2;
  const project = (point) => ({
    x: xOffset + ((Number(point.x) || 0) - minX) * scale,
    y: height - (yOffset + ((Number(point.y) || 0) - minY) * scale),
  });
  const cellMarkup = (() => {
    if (preview.boundary !== 'periodic' || !cell || !cell.a || !cell.b) {
      return '';
    }
    const origin = cell.origin || { x: 0, y: 0 };
    const a = cell.a;
    const b = cell.b;
    const corners = [
      origin,
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) + (Number(b.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(b.y) || 0) },
    ].map((point) => project(point));
    return `<polygon class="cell-boundary" points="${corners.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')}" />`;
  })();
  const edgeMarkup = bonds.map((bond) => {
    const source = siteMap.get(String(bond.source));
    const targetSite = siteMap.get(String(bond.target));
    if (!source || !targetSite) {
      return '';
    }
    const p1 = project(source);
    const p2 = project(targetSite);
    if (bond.periodic) {
      const offset = bond.offset || { x: 0, y: 0 };
      const targetImage = { x: targetSite.x + (Number(offset.x) || 0), y: targetSite.y + (Number(offset.y) || 0) };
      const dx = targetImage.x - source.x;
      const dy = targetImage.y - source.y;
      const length = Math.max(Math.hypot(dx, dy), 1e-8);
      const unit = { x: dx / length, y: dy / length };
      const halfLength = Math.min(38, Math.max(20, length * scale * 0.42));
      const sourceEnd = { x: p1.x + unit.x * halfLength, y: p1.y - unit.y * halfLength };
      const targetEnd = { x: p2.x - unit.x * halfLength, y: p2.y + unit.y * halfLength };
      return [
        `<line x1="${p1.x.toFixed(2)}" y1="${p1.y.toFixed(2)}" x2="${sourceEnd.x.toFixed(2)}" y2="${sourceEnd.y.toFixed(2)}" stroke="#475569" stroke-width="3.2" stroke-linecap="round" />`,
        `<line x1="${p2.x.toFixed(2)}" y1="${p2.y.toFixed(2)}" x2="${targetEnd.x.toFixed(2)}" y2="${targetEnd.y.toFixed(2)}" stroke="#475569" stroke-width="3.2" stroke-linecap="round" />`,
      ].join('');
    }
    return `<line x1="${p1.x.toFixed(2)}" y1="${p1.y.toFixed(2)}" x2="${p2.x.toFixed(2)}" y2="${p2.y.toFixed(2)}" stroke="#475569" stroke-width="3.2" stroke-linecap="round" />`;
  }).join('');
  const ghostMarkup = bonds.filter((bond) => bond.periodic).map((bond) => {
    const source = siteMap.get(String(bond.source));
    const targetSite = siteMap.get(String(bond.target));
    if (!source || !targetSite) {
      return '';
    }
    const offset = bond.offset || { x: 0, y: 0 };
    const dx = Number(offset.x) || 0;
    const dy = Number(offset.y) || 0;
    const sourceMirror = project({ x: source.x - dx, y: source.y - dy });
    const targetMirror = project({ x: targetSite.x + dx, y: targetSite.y + dy });
    const sourcePoint = project(source);
    const targetPoint = project(targetSite);
    const ghostSite = (point, label) => [
      `<circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="12" class="ghost-site" />`,
      `<text x="${point.x.toFixed(2)}" y="${(point.y + 4).toFixed(2)}" class="ghost-site-label">${escapeHtml(label)}</text>`,
    ].join('');
    return [
      `<line x1="${sourcePoint.x.toFixed(2)}" y1="${sourcePoint.y.toFixed(2)}" x2="${targetMirror.x.toFixed(2)}" y2="${targetMirror.y.toFixed(2)}" class="ghost-bond" />`,
      `<line x1="${sourceMirror.x.toFixed(2)}" y1="${sourceMirror.y.toFixed(2)}" x2="${targetPoint.x.toFixed(2)}" y2="${targetPoint.y.toFixed(2)}" class="ghost-bond" />`,
      ghostSite(sourceMirror, source.id),
      ghostSite(targetMirror, targetSite.id),
    ].join('');
  }).join('');
  const siteMarkup = numericSites.map((site) => {
    const p = project(site);
    return [
      `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="17" fill="#ffffff" stroke="#334155" stroke-width="3" />`,
      `<text x="${p.x.toFixed(2)}" y="${(p.y + 5).toFixed(2)}" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">${escapeHtml(site.id)}</text>`,
    ].join('');
  }).join('');
  const meta = [
    `${langText('model', '模型')}: ${preview.model || ''}`,
    `${langText('representation', '表示')}: ${preview.representation || 'finite_cluster'}`,
    `${langText('solver', '求解器')}: ${preview.solver || ''}`,
    `${langText('dimension', '维度')}: ${preview.dimension || ''}D`,
    `${langText('preset', '模板')}: ${preview.preset || ''}`,
    `${langText('boundary', '边界')}: ${preview.boundary || ''}`,
    `nelec: ${Array.isArray(preview.nelec) ? `(${preview.nelec.join(', ')})` : ''}`,
    `${langText('electrons/cell', '每原胞电子数')}: ${preview.occupation && preview.occupation.electrons_per_cell !== undefined ? preview.occupation.electrons_per_cell : ''}`,
    `k-mesh: ${preview.reciprocal_space && Array.isArray(preview.reciprocal_space.kmesh) ? preview.reciprocal_space.kmesh.join(' x ') : ''}`,
    `sites: ${preview.site_count ?? 0}`,
    `bonds: ${preview.bond_count ?? 0}`,
  ].filter((item) => !item.endsWith(': '));
  target.innerHTML = [
    `<div class="model-structure-meta">${meta.map((item) => `<span class="model-structure-pill">${escapeHtml(item)}</span>`).join('')}</div>`,
    `<svg class="model-structure-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(t('modelHamiltonianPreviewTitle'))}">`,
    '<g>',
    cellMarkup,
    ghostMarkup,
    edgeMarkup,
    siteMarkup,
    '</g>',
    '</svg>',
    preview.truncated ? `<div class="model-structure-truncated">${escapeHtml(t('modelHamiltonianPreviewTruncated'))}</div>` : '',
  ].join('');
}

async function previewModelHamiltonianStructure() {
  const inputFile = document.getElementById('model-hamiltonian-input-file').value.trim();
  if (!inputFile) {
    resetModelHamiltonianStructurePreview(t('modelHamiltonianMissingFile'));
    return;
  }
  resetModelHamiltonianStructurePreview(t('modelHamiltonianPreviewLoading'));
  try {
    const response = await fetch('/api/model-hamiltonian-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_hamiltonian_input_file: inputFile,
        solver: document.getElementById('model-solver').value === 'dmet'
          ? { name: 'dmet', options: collectDmetSolverOptions() }
          : document.getElementById('model-solver').value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `${t('modelHamiltonianPreviewFailed')}: HTTP ${response.status}`);
    }
    lastModelHamiltonianPreview = payload.preview;
    if (payload.preview && payload.preview.representation === 'bloch' && payload.preview.solver) {
      setSelectValue('model-solver', payload.preview.solver);
    }
    renderModelHamiltonianStructurePreview(lastModelHamiltonianPreview);
    renderDmetSpinSector();
    snapshotCurrentTaskSession();
    return lastModelHamiltonianPreview;
  } catch (error) {
    resetModelHamiltonianStructurePreview(`${t('modelHamiltonianPreviewFailed')}: ${error instanceof Error ? error.message : String(error)}`);
    snapshotCurrentTaskSession();
    return null;
  }
}

function resetPeriodicStructurePreview(message) {
  lastPeriodicStructurePreview = null;
  const preview = document.getElementById('periodic-structure-preview');
  if (preview) {
    preview.textContent = message || t('periodicPreviewEmpty');
  }
}

function formatPeriodicNumberList(values, digits = 4) {
  if (!Array.isArray(values)) {
    return '';
  }
  return values.map((value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(digits) : String(value);
  }).join(', ');
}

function normalizePeriodicBandPath(value) {
  return String(value || '')
    .trim()
    .replace(/[Γγ]/g, 'G')
    .replace(/[\s\-→–—]+/g, '');
}

function formatPeriodicBandSpecialPoints(points) {
  if (!points || typeof points !== 'object' || Array.isArray(points)) {
    return '';
  }
  return Object.entries(points)
    .map(([label, coordinates]) => `${label} = ${formatPeriodicNumberList(coordinates, 8)}`)
    .join('\n');
}

function parsePeriodicBandSpecialPoints(value) {
  const points = {};
  const lines = String(value || '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length > 64) {
    return { points, error: langText('At most 64 explicit special points are allowed.', '最多允许 64 个显式高对称点。') };
  }
  for (const line of lines) {
    const match = line.match(/^([^:=]+)\s*[:=]\s*(.+)$/);
    if (!match) {
      return {
        points,
        error: langText(
          `Invalid special-point row: ${line}`,
          `高对称点坐标行格式无效：${line}`
        ),
      };
    }
    const label = match[1].trim().replace(/[Γγ]/g, 'G');
    if (!/^[A-Z][a-z0-9]*$/.test(label)) {
      return {
        points,
        error: langText(`Invalid special-point label: ${label}`, `高对称点标签无效：${label}`),
      };
    }
    if (Object.prototype.hasOwnProperty.call(points, label)) {
      return {
        points,
        error: langText(`Duplicate special-point label: ${label}`, `高对称点标签重复：${label}`),
      };
    }
    const coordinates = match[2].split(/[\s,]+/).filter(Boolean).map(Number);
    if (coordinates.length !== 3 || coordinates.some((coordinate) => !Number.isFinite(coordinate))) {
      return {
        points,
        error: langText(
          `${label} requires three finite reduced reciprocal coordinates.`,
          `${label} 必须包含三个有限的约化倒空间坐标。`
        ),
      };
    }
    points[label] = coordinates;
  }
  return { points, error: '' };
}

function seedPeriodicBandPathFromPreview() {
  const mode = document.getElementById('periodic-band-path-mode').value;
  const preview = lastPeriodicStructurePreview && lastPeriodicStructurePreview.band_path_preview;
  if (mode === 'auto' || mode === 'seekpath' || !preview || preview.status !== 'available') {
    return;
  }
  const pathInput = document.getElementById('periodic-band-path');
  if (!pathInput.value.trim() && preview.automatic_path) {
    pathInput.value = preview.automatic_path;
  }
  const specialPointInput = document.getElementById('periodic-band-special-points');
  if (mode === 'explicit' && !specialPointInput.value.trim()) {
    specialPointInput.value = formatPeriodicBandSpecialPoints(preview.special_points_scaled);
  }
}

function syncPeriodicBandPathControls(seedFromPreview = false) {
  const mode = document.getElementById('periodic-band-path-mode').value;
  const seekpathMode = mode === 'seekpath';
  const labeledPathMode = mode === 'custom' || mode === 'explicit';
  document.getElementById('periodic-band-path-npoints-control').classList.toggle('hidden', seekpathMode);
  document.getElementById('periodic-band-path-reference-distance-control').classList.toggle('hidden', !seekpathMode);
  document.getElementById('periodic-band-path-symprec-control').classList.toggle('hidden', !seekpathMode);
  document.getElementById('periodic-band-path-control').classList.toggle('hidden', !labeledPathMode);
  document.getElementById('periodic-band-special-points-control').classList.toggle('hidden', mode !== 'explicit');
  if (seedFromPreview) {
    seedPeriodicBandPathFromPreview();
  }
}

function renderPeriodicStructurePreview(preview) {
  const target = document.getElementById('periodic-structure-preview');
  if (!target) {
    return;
  }
  if (!preview || typeof preview !== 'object') {
    target.textContent = t('periodicPreviewEmpty');
    return;
  }
  const composition = preview.composition && typeof preview.composition === 'object'
    ? Object.entries(preview.composition).map(([element, count]) => `${element}: ${count}`).join(', ')
    : '';
  const volume = Number(preview.cell_volume_angstrom3);
  const items = [
    [langText('Formula', '化学式'), preview.formula],
    [langText('Atoms', '原子数'), preview.atom_count],
    [langText('Source sites', '输入独立位点数'), preview.source_site_count],
    [langText('Symmetry expansion', '对称性展开'), preview.symmetry_expanded ? langText('expanded to P1', '已展开为 P1') : langText('not required', '无需展开')],
    [langText('Space group', '空间群'), preview.space_group_symbol
      ? `${preview.space_group_symbol}${preview.space_group_number ? ` (#${preview.space_group_number})` : ''}`
      : ''],
    [langText('Parser', '解析器'), preview.parser],
    [langText('Composition', '元素组成'), composition],
    [langText('Format', '格式'), String(preview.source_format || '').toUpperCase()],
    [langText('Lattice lengths (Angstrom)', '晶格长度 (Angstrom)'), formatPeriodicNumberList(preview.lattice_lengths_angstrom)],
    [langText('Lattice angles (degree)', '晶格角 (degree)'), formatPeriodicNumberList(preview.lattice_angles_degree, 2)],
    [langText('Cell volume (Angstrom^3)', '晶胞体积 (Angstrom^3)'), Number.isFinite(volume) ? volume.toFixed(4) : ''],
  ].filter((item) => item[1] !== null && typeof item[1] !== 'undefined' && String(item[1]).trim());
  const bandPreview = preview.band_path_preview && typeof preview.band_path_preview === 'object'
    ? preview.band_path_preview
    : null;
  let bandPreviewHtml = '';
  if (bandPreview && bandPreview.status === 'available') {
    const specialPoints = bandPreview.special_points_scaled && typeof bandPreview.special_points_scaled === 'object'
      ? Object.entries(bandPreview.special_points_scaled)
      : [];
    const pointRows = specialPoints.map(([label, coordinates]) => [
      '<tr>',
      `<td><strong>${escapeHtml(label)}</strong></td>`,
      `<td>${escapeHtml(formatPeriodicNumberList(coordinates, 6))}</td>`,
      '</tr>',
    ].join('')).join('');
    bandPreviewHtml = [
      '<div class="periodic-band-preview">',
      '<div class="periodic-band-preview-meta">',
      '<div class="periodic-preview-item">',
      `<span>${escapeHtml(langText('Bravais lattice', 'Bravais 晶格'))}</span>`,
      `<strong>${escapeHtml(bandPreview.bravais_lattice || bandPreview.bravais_lattice_name || '')}</strong>`,
      '</div>',
      '<div class="periodic-preview-item">',
      `<span>${escapeHtml(langText('Automatic path', '自动路径'))}</span>`,
      `<strong>${escapeHtml(bandPreview.automatic_path || '')}</strong>`,
      '</div>',
      '</div>',
      pointRows ? [
        '<table class="periodic-special-points">',
        '<thead><tr>',
        `<th>${escapeHtml(langText('Point', '高对称点'))}</th>`,
        `<th>${escapeHtml(langText('Reduced reciprocal coordinates', '约化倒空间坐标'))}</th>`,
        '</tr></thead>',
        `<tbody>${pointRows}</tbody>`,
        '</table>',
      ].join('') : '',
      '</div>',
    ].join('');
  } else if (bandPreview && bandPreview.reason) {
    bandPreviewHtml = `<div class="periodic-band-preview muted">${escapeHtml(bandPreview.reason)}</div>`;
  }
  const seekpathPreview = preview.seekpath_preview && typeof preview.seekpath_preview === 'object'
    ? preview.seekpath_preview
    : null;
  let seekpathPreviewHtml = '';
  if (seekpathPreview && seekpathPreview.status === 'available') {
    const volumeRatio = Number(seekpathPreview.volume_original_wrt_primitive);
    const warningText = Array.isArray(seekpathPreview.warnings)
      ? seekpathPreview.warnings.map((warning) => warning && warning.message).filter(Boolean).join(' | ')
      : '';
    const seekpathItems = [
      [langText('Space group', '空间群'), `${seekpathPreview.spacegroup_international || ''} (#${seekpathPreview.spacegroup_number || ''})`],
      [langText('Bravais lattice', 'Bravais 晶格'), seekpathPreview.bravais_lattice_extended || seekpathPreview.bravais_lattice],
      [langText('Primitive atoms', '标准原胞原子数'), seekpathPreview.primitive_atom_count],
      [langText('Input / primitive volume', '输入/标准原胞体积比'), Number.isFinite(volumeRatio) ? volumeRatio.toFixed(4) : ''],
      [langText('Generated path points', '生成路径点数'), seekpathPreview.explicit_point_count],
      [langText('Standard path', '标准路径'), seekpathPreview.path],
    ].filter((item) => item[1] !== null && typeof item[1] !== 'undefined' && String(item[1]).trim());
    seekpathPreviewHtml = [
      '<div class="periodic-band-preview">',
      `<strong>${escapeHtml(langText('SeeK-path standardized primitive cell', 'SeeK-path 标准原胞'))}</strong>`,
      '<div class="periodic-band-preview-meta">',
      seekpathItems.map(([label, value]) => [
        '<div class="periodic-preview-item">',
        `<span>${escapeHtml(label)}</span>`,
        `<strong>${escapeHtml(value)}</strong>`,
        '</div>',
      ].join('')).join(''),
      '</div>',
      warningText ? `<div class="muted">${escapeHtml(warningText)}</div>` : '',
      '</div>',
    ].join('');
  } else if (seekpathPreview && seekpathPreview.reason) {
    seekpathPreviewHtml = `<div class="periodic-band-preview muted">${escapeHtml(seekpathPreview.reason)}</div>`;
  }
  target.innerHTML = `<div class="periodic-preview-grid">${items.map(([label, value]) => [
    '<div class="periodic-preview-item">',
    `<span>${escapeHtml(label)}</span>`,
    `<strong>${escapeHtml(value)}</strong>`,
    '</div>',
  ].join('')).join('')}</div>${bandPreviewHtml}${seekpathPreviewHtml}`;
}

async function previewPeriodicStructure() {
  const structureText = document.getElementById('periodic-structure-text').value.trim();
  const structureFormat = document.getElementById('periodic-format').value;
  const seekpathMode = document.getElementById('periodic-band-path-mode').value === 'seekpath';
  if (!structureText) {
    resetPeriodicStructurePreview(t('periodicMissingStructure'));
    snapshotCurrentTaskSession();
    return false;
  }
  resetPeriodicStructurePreview(t('periodicPreviewLoading'));
  try {
    const response = await fetch('/api/periodic-structure-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        structure_text: structureText,
        structure_format: structureFormat,
        seekpath_symprec: numericInputValue('periodic-band-path-symprec', 1e-5),
        seekpath_reference_distance: seekpathMode
          ? numericInputValue('periodic-band-path-reference-distance', 0.025)
          : null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `${t('periodicPreviewFailed')}: HTTP ${response.status}`);
    }
    lastPeriodicStructurePreview = payload.preview;
    renderPeriodicStructurePreview(lastPeriodicStructurePreview);
    seedPeriodicBandPathFromPreview();
    snapshotCurrentTaskSession();
    return true;
  } catch (error) {
    resetPeriodicStructurePreview(`${t('periodicPreviewFailed')}: ${error instanceof Error ? error.message : String(error)}`);
    snapshotCurrentTaskSession();
    return false;
  }
}

async function handlePeriodicFile(event) {
  const files = event && event.target ? event.target.files : null;
  const file = files && files.length ? files[0] : null;
  if (!file) {
    return;
  }
  try {
    const filename = String(file.name || '').toLowerCase();
    if (filename.endsWith('.cif')) {
      document.getElementById('periodic-format').value = 'cif';
    } else if (filename.endsWith('.vasp') || filename.endsWith('.poscar') || filename === 'poscar' || filename === 'contcar') {
      document.getElementById('periodic-format').value = 'poscar';
    }
    document.getElementById('periodic-structure-text').value = await file.text();
    invalidatePendingExecution();
    resetPeriodicStructurePreview();
    snapshotCurrentTaskSession();
    await previewPeriodicStructure();
  } catch (error) {
    resetPeriodicStructurePreview(`${t('periodicPreviewFailed')}: ${error instanceof Error ? error.message : String(error)}`);
    snapshotCurrentTaskSession();
  }
}

function syncPeriodicMethodControls() {
  const isDft = document.getElementById('periodic-method').value === 'dft';
  document.getElementById('periodic-xc-control').classList.toggle('hidden', !isDft);
  document.getElementById('periodic-xc').disabled = !isDft;
}

function syncPeriodicCorrelationControls() {
  const correlationTreatment = document.getElementById('periodic-correlation-treatment').value;
  const usesGw = correlationTreatment === 'gw' || correlationTreatment === 'gw_dmft';
  const usesDmft = correlationTreatment === 'hf_dmft' || correlationTreatment === 'gw_dmft';
  const gwControls = document.getElementById('periodic-gw-controls');
  gwControls.classList.toggle('hidden', !usesGw);
  gwControls.querySelectorAll('input, select').forEach((item) => {
    item.disabled = !usesGw;
  });
  const controls = document.getElementById('periodic-dmft-controls');
  controls.classList.toggle('hidden', !usesDmft);
  controls.querySelectorAll('input, select').forEach((item) => {
    item.disabled = !usesDmft;
  });
  document.getElementById('periodic-dmft-controls-title').textContent = correlationTreatment === 'gw_dmft'
    ? 'GW+DMFT'
    : 'HF+DMFT';

  const method = document.getElementById('periodic-method');
  method.disabled = correlationTreatment !== 'none';
  if (correlationTreatment !== 'none') {
    method.value = usesGw ? 'dft' : 'hf';
    document.getElementById('periodic-smearing').value = 'none';
    document.getElementById('periodic-density-fitting').value = 'gdf';
    document.getElementById('periodic-kpoint-scheme').value = 'gamma_centered';
    ['x', 'y', 'z'].forEach((axis) => {
      document.getElementById(`periodic-kpoint-shift-${axis}`).value = '0';
    });
    if (usesGw) {
      document.getElementById('periodic-restricted').value = 'true';
    }
    if (usesDmft) {
      const restricted = document.getElementById('periodic-restricted').value;
      const impuritySolver = document.getElementById('periodic-dmft-impurity-solver');
      if (correlationTreatment === 'gw_dmft' && impuritySolver.value === 'ucc') {
        impuritySolver.value = 'cc';
      } else if (impuritySolver.value !== 'fci') {
        impuritySolver.value = restricted === 'false' ? 'ucc' : 'cc';
      }
    }
  }
  [
    'periodic-kpoint-scheme',
    'periodic-kpoint-shift-x',
    'periodic-kpoint-shift-y',
    'periodic-kpoint-shift-z',
    'periodic-smearing',
  ].forEach((id) => {
    document.getElementById(id).disabled = correlationTreatment !== 'none';
  });
  syncPeriodicMethodControls();
  syncPeriodicNumericalControls();
}

function syncPeriodicNumericalControls() {
  const densityFitting = document.getElementById('periodic-density-fitting').value;
  const supportsAuxbasis = densityFitting === 'gdf' || densityFitting === 'mdf';
  document.getElementById('periodic-df-auxbasis').disabled = !supportsAuxbasis;
  const smearingEnabled = document.getElementById('periodic-smearing').value !== 'none';
  document.getElementById('periodic-smearing-sigma').disabled = !smearingEnabled;
  document.getElementById('periodic-smearing-fix-spin').disabled = !smearingEnabled;
}

function syncModelSolverControls() {
  const solver = document.getElementById('model-solver').value;
  const supportsRoots = solver === 'fci' || solver === 'block2_dmrg';
  const usesDmet = solver === 'dmet';
  document.getElementById('model-root-control').classList.toggle('hidden', !supportsRoots);
  document.getElementById('model-nroots').disabled = !supportsRoots;
  document.getElementById('model-dmet-controls').classList.toggle('hidden', !usesDmet);
  [
    'model-dmet-execution-mode',
    'model-dmet-fragment-mode',
    'model-dmet-impurity-shape',
    'model-dmet-impurity-size',
    'model-dmet-impurity-solver',
    'model-dmet-block2-preset',
    'model-dmet-block2-ordering',
    'model-dmet-reference',
    'model-dmet-reference-density',
    'model-dmet-interacting-bath',
    'model-dmet-max-iterations',
    'model-dmet-energy-tolerance',
    'model-dmet-density-tolerance',
  ].forEach((id) => {
    document.getElementById(id).disabled = !usesDmet;
  });
  syncDmetImpuritySolverControls();
  syncDmetFragmentControls();
  const excitedStatesOutput = document.querySelector(
    '#model-output-group input[name="outputs"][value="excited_states"]'
  );
  if (excitedStatesOutput) {
    excitedStatesOutput.disabled = !supportsRoots;
    if (!supportsRoots) excitedStatesOutput.checked = false;
  }
  renderDmetSpinSector();
}

function syncDmetImpuritySolverControls() {
  const usesDmet = document.getElementById('model-solver').value === 'dmet';
  const usesCcsd = usesDmet
    && document.getElementById('model-dmet-impurity-solver').value === 'ccsd';
  document.getElementById('model-dmet-ccsd-beta-control').classList.toggle('hidden', !usesCcsd);
  document.getElementById('model-dmet-ccsd-beta').disabled = !usesCcsd;
  const usesBlock2 = usesDmet
    && document.getElementById('model-dmet-impurity-solver').value === 'block2_dmrg';
  [
    ['model-dmet-block2-preset-control', 'model-dmet-block2-preset'],
    ['model-dmet-block2-ordering-control', 'model-dmet-block2-ordering'],
  ].forEach(([controlId, inputId]) => {
    document.getElementById(controlId).classList.toggle('hidden', !usesBlock2);
    document.getElementById(inputId).disabled = !usesBlock2;
  });
}

function syncDmetFragmentControls() {
  const executionMode = document.getElementById('model-dmet-execution-mode').value;
  const fragmentModeControl = document.getElementById('model-dmet-fragment-mode');
  const siteCountOption = fragmentModeControl.querySelector('option[value="site_count"]');
  if (siteCountOption) siteCountOption.disabled = executionMode === 'translational';
  if (executionMode === 'translational' && fragmentModeControl.value === 'site_count') {
    fragmentModeControl.value = 'primitive_cell';
  }
  const mode = fragmentModeControl.value;
  const usesDmet = document.getElementById('model-solver').value === 'dmet';
  const shapeControl = document.getElementById('model-dmet-impurity-shape-control');
  const sizeControl = document.getElementById('model-dmet-impurity-size-control');
  const shapeInput = document.getElementById('model-dmet-impurity-shape');
  const sizeInput = document.getElementById('model-dmet-impurity-size');
  shapeControl.classList.toggle('hidden', mode !== 'cell_shape');
  sizeControl.classList.toggle('hidden', mode !== 'site_count');
  shapeInput.disabled = !usesDmet || mode !== 'cell_shape';
  sizeInput.disabled = !usesDmet || mode !== 'site_count';
  shapeInput.required = usesDmet && mode === 'cell_shape';
  sizeInput.required = usesDmet && mode === 'site_count';
}

function syncTaskFamilyControls() {
  const taskFamily = currentTaskFamily();
  const isModelHamiltonian = taskFamily === 'model_hamiltonian';
  const isPeriodic = taskFamily === 'periodic';
  const isMolecular = taskFamily === 'molecular';
  document.getElementById('model-hamiltonian-section').classList.toggle('hidden', !isModelHamiltonian);
  document.getElementById('periodic-section').classList.toggle('hidden', !isPeriodic);
  document.getElementById('molecular-section').classList.toggle('hidden', !isMolecular);
  document.getElementById('molecular-output-group').classList.toggle('hidden', !isMolecular);
  document.getElementById('model-output-group').classList.toggle('hidden', !isModelHamiltonian);
  document.getElementById('periodic-output-group').classList.toggle('hidden', !isPeriodic);
  const taskFamilySelect = document.getElementById('task-family');
  const taskFamilyOption = taskFamilySelect.options[taskFamilySelect.selectedIndex];
  const taskFamilyText = taskFamilyOption ? taskFamilyOption.textContent : '';
  taskFamilySelect.title = taskFamilyText;
  const taskFamilyDetail = document.getElementById('task-family-effective');
  if (taskFamilyDetail) {
    taskFamilyDetail.textContent = taskFamilyText;
  }
  updateEffectiveWorkDirLabel();
  syncModelSolverControls();
  syncMethodControls();
  syncPeriodicMethodControls();
  syncPeriodicCorrelationControls();
}

function openModelHamiltonianBuilder() {
  invalidatePendingExecution();
  window.name = 'pyscf-agent-web-ui';
  const runId = ensureCurrentRunId();
  const params = new URLSearchParams({
    work_dir: builderWorkDirValue(),
    run_id: runId,
    target: 'agent',
  });
  window.open(`/model-hamiltonian-builder/?${params.toString()}`, 'pyscf-agent-model-builder');
}

function handleBuilderMessage(event) {
  if (event.origin !== window.location.origin || !event.data || event.data.type !== 'pyscf-agent:model-hamiltonian-input-saved') {
    return;
  }
  if (event.data.target && event.data.target !== 'agent') {
    return;
  }
  const incomingPath = typeof event.data.path === 'string' ? event.data.path.trim() : '';
  const incomingWorkDir = typeof event.data.work_dir === 'string' ? event.data.work_dir.trim() : '';
  const incomingRunId = typeof event.data.run_id === 'string' ? event.data.run_id.trim() : '';
  const incomingSolver = typeof event.data.solver === 'string' ? event.data.solver.trim() : '';
  const incomingRunDir = typeof event.data.run_dir === 'string' && event.data.run_dir.trim()
    ? event.data.run_dir.trim()
    : directoryFromPath(incomingPath);
  if (currentTaskFamily() !== 'model_hamiltonian' || (lastExecutionPayload && currentTaskHasMeaningfulState())) {
    createTaskSession('model_hamiltonian', { inheritWorkDir: true });
  }
  document.getElementById('task-family').value = 'model_hamiltonian';
  if (incomingSolver) {
    setSelectValue('model-solver', incomingSolver);
  }
  if (typeof event.data.path === 'string' && event.data.path.trim()) {
    document.getElementById('model-hamiltonian-input-file').value = incomingPath;
  }
  if (incomingRunDir) {
    document.getElementById('work-dir').value = incomingRunDir;
    setWorkDirLocked(true);
  } else if (incomingWorkDir) {
    document.getElementById('work-dir').value = incomingWorkDir;
    setWorkDirLocked(false);
  }
  if (incomingRunId) {
    currentRunId = incomingRunId;
  }
  syncTaskFamilyControls();
  invalidatePendingExecution();
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('builderInputLinked') }]);
  renderConversation(conversationHistory);
  snapshotCurrentTaskSession();
  previewModelHamiltonianStructure();
}

function parsedDmetImpurityShape() {
  const raw = document.getElementById('model-dmet-impurity-shape').value.trim();
  if (!raw) return [];
  const parts = raw.split(/[,xX\s]+/).filter(Boolean);
  if (!parts.length || parts.some((value) => !/^\d+$/.test(value) || Number(value) < 1)) {
    return null;
  }
  return parts.map((value) => Number.parseInt(value, 10));
}

function collectDmetSolverOptions() {
  const executionMode = document.getElementById('model-dmet-execution-mode').value;
  const fragmentMode = document.getElementById('model-dmet-fragment-mode').value;
  const impurityShape = parsedDmetImpurityShape();
  const impuritySizeValue = document.getElementById('model-dmet-impurity-size').value.trim();
  const impuritySize = /^\d+$/.test(impuritySizeValue) && Number(impuritySizeValue) > 0
    ? Number.parseInt(impuritySizeValue, 10)
    : null;
  const impuritySolver = document.getElementById('model-dmet-impurity-solver').value;
  const options = {
    impurity_solver: impuritySolver,
    impurity_solver_options: impuritySolver === 'block2_dmrg'
      ? {
        preset: document.getElementById('model-dmet-block2-preset').value,
        orbital_ordering: document.getElementById('model-dmet-block2-ordering').value,
      }
      : {},
    impurity_shape: fragmentMode === 'cell_shape' && Array.isArray(impurityShape)
      ? impurityShape
      : [],
    impurity_size: fragmentMode === 'site_count' ? impuritySize : null,
    reference: document.getElementById('model-dmet-reference').value,
    reference_density_guess: document.getElementById('model-dmet-reference-density').value,
    interacting_bath: document.getElementById('model-dmet-interacting-bath').value !== 'false',
    max_iterations: Math.max(1, integerInputValue('model-dmet-max-iterations', 50)),
    energy_tolerance: numericInputValue('model-dmet-energy-tolerance', 1e-6),
    density_tolerance: numericInputValue('model-dmet-density-tolerance', 1e-4),
  };
  const betaInput = document.getElementById('model-dmet-ccsd-beta');
  if (impuritySolver === 'ccsd' && (betaInput.value.trim() || betaInput.validity.badInput)) {
    options.impurity_solver_options.beta = betaInput.validity.badInput ? null : Number(betaInput.value);
  }
  if (executionMode) options.execution_mode = executionMode;
  if (fragmentMode) options.fragment_definition = fragmentMode;
  return options;
}

async function validateDmetModelControls() {
  if (document.getElementById('model-solver').value !== 'dmet') return '';
  const executionMode = document.getElementById('model-dmet-execution-mode').value;
  const fragmentMode = document.getElementById('model-dmet-fragment-mode').value;
  if (executionMode === 'translational' && fragmentMode === 'honeycomb_hexagon') {
    return 'Honeycomb hexagons require Automatic or Finite graph execution mode.';
  }
  if (executionMode === 'translational' && fragmentMode === 'site_count') {
    return 'Translated representative DMET requires a primitive-cell or cell-block fragment definition.';
  }
  if (fragmentMode === 'cell_shape') {
    const impurityShape = parsedDmetImpurityShape();
    if (!Array.isArray(impurityShape) || !impurityShape.length) {
      return 'Fragment Shape must contain positive primitive-cell counts, for example 2 or 2,2.';
    }
  }
  if (fragmentMode === 'site_count') {
    const impuritySize = document.getElementById('model-dmet-impurity-size').value.trim();
    if (!/^\d+$/.test(impuritySize) || Number(impuritySize) < 1) {
      return 'Impurity Size must be a positive site count.';
    }
  }
  const preview = await previewModelHamiltonianStructure();
  const validation = preview && preview.dmet_validation && typeof preview.dmet_validation === 'object'
    ? preview.dmet_validation
    : null;
  if (!validation) {
    return 'DMET compatibility could not be checked. Review the model input and preview it again.';
  }
  const errors = Array.isArray(validation.errors) ? validation.errors.filter(Boolean) : [];
  return validation.valid && !errors.length
    ? ''
    : (errors.join(' | ') || 'The selected DMET settings are not compatible with this model.');
}

function collectModelHamiltonianFormData() {
  const solverName = document.getElementById('model-solver').value;
  const supportsRoots = solverName === 'fci' || solverName === 'block2_dmrg';
  const usesDmet = solverName === 'dmet';
  const outputs = selectedOutputValues().filter(
    (output) => supportsRoots || output !== 'excited_states'
  );
  const nroots = Math.max(1, integerInputValue('model-nroots', 1));
  if (supportsRoots && nroots > 1 && !outputs.includes('excited_states')) {
    outputs.push('excited_states');
  }
  let solver = solverName;
  if (supportsRoots) {
    solver = { name: solverName, options: { nroots } };
  } else if (usesDmet) {
    solver = {
      name: 'dmet',
      options: collectDmetSolverOptions(),
    };
  }
  return {
    task_type: 'model_hamiltonian',
    model_hamiltonian_input_file: document.getElementById('model-hamiltonian-input-file').value.trim(),
    solver,
    outputs,
  };
}

function collectPeriodicFormData() {
  const method = document.getElementById('periodic-method').value;
  const correlationTreatment = document.getElementById('periodic-correlation-treatment').value;
  const smearingMethod = document.getElementById('periodic-smearing').value;
  const densityFittingMethod = document.getElementById('periodic-density-fitting').value;
  const bandPathMode = document.getElementById('periodic-band-path-mode').value;
  const bandPath = normalizePeriodicBandPath(document.getElementById('periodic-band-path').value);
  const explicitSpecialPoints = parsePeriodicBandSpecialPoints(
    document.getElementById('periodic-band-special-points').value
  ).points;
  const payload = {
    task_type: 'periodic',
    periodic: {
      structure_format: document.getElementById('periodic-format').value,
      structure_text: document.getElementById('periodic-structure-text').value.trim(),
      basis: document.getElementById('periodic-basis').value,
      pseudo: document.getElementById('periodic-pseudo').value,
      kmesh: [
        integerInputValue('periodic-kmesh-x', 1),
        integerInputValue('periodic-kmesh-y', 1),
        integerInputValue('periodic-kmesh-z', 1),
      ],
      kpoint_scheme: document.getElementById('periodic-kpoint-scheme').value,
      band_path_mode: bandPathMode,
      band_path: bandPathMode === 'custom' || bandPathMode === 'explicit' ? bandPath : null,
      band_path_special_points: bandPathMode === 'explicit' ? explicitSpecialPoints : {},
      band_path_npoints: integerInputValue('periodic-band-path-npoints', 80),
      band_path_reference_distance: numericInputValue('periodic-band-path-reference-distance', 0.025),
      band_path_symprec: numericInputValue('periodic-band-path-symprec', 1e-5),
      kpoint_shift: [
        numericInputValue('periodic-kpoint-shift-x', 0),
        numericInputValue('periodic-kpoint-shift-y', 0),
        numericInputValue('periodic-kpoint-shift-z', 0),
      ],
      dimension: 3,
      precision: numericInputValue('periodic-precision', 1e-8),
      ke_cutoff: optionalNumericInputValue('periodic-ke-cutoff'),
      fft_mesh: optionalIntegerTriplet([
        'periodic-fft-mesh-x',
        'periodic-fft-mesh-y',
        'periodic-fft-mesh-z',
      ]),
      density_fitting_method: densityFittingMethod,
      density_fitting_auxbasis: densityFittingMethod === 'gdf' || densityFittingMethod === 'mdf'
        ? (document.getElementById('periodic-df-auxbasis').value.trim() || null)
        : null,
      exxdiv: document.getElementById('periodic-exxdiv').value,
      smearing_method: smearingMethod,
      smearing_sigma: smearingMethod === 'none'
        ? null
        : optionalNumericInputValue('periodic-smearing-sigma'),
      smearing_fix_spin: smearingMethod !== 'none'
        && document.getElementById('periodic-smearing-fix-spin').checked,
    },
    method,
    xc: method === 'dft' ? document.getElementById('periodic-xc').value : null,
    job: 'single_point',
    charge: integerInputValue('periodic-charge', 0),
    spin: integerInputValue('periodic-spin', 0),
    outputs: selectedOutputValues(),
  };
  const restricted = document.getElementById('periodic-restricted').value;
  if (restricted !== 'auto') {
    payload.restricted = restricted === 'true';
  }
  const usesGw = correlationTreatment === 'gw' || correlationTreatment === 'gw_dmft';
  const usesDmft = correlationTreatment === 'hf_dmft' || correlationTreatment === 'gw_dmft';
  const gwOptions = usesGw ? {
    gw_analytic_continuation: 'pade',
    gw_broadening: numericInputValue('periodic-gw-broadening', 0.1) / 27.211386,
    gw_frequency_window: [
      0,
      numericInputValue('periodic-gw-frequency-maximum', 18) / 27.211386,
    ],
    gw_real_frequency_points: Math.max(2, integerInputValue('periodic-gw-real-points', 181)),
    gw_imaginary_frequency_points: Math.max(1, integerInputValue('periodic-gw-imaginary-points', 100)),
    gw_full_self_energy: true,
    gw_finite_size_correction: document.getElementById('periodic-gw-finite-size-correction').checked,
    gw_quasiparticle_energies: true,
  } : {};
  if (correlationTreatment === 'gw') {
    payload.method = 'dft';
    payload.restricted = true;
    payload.solver = { name: 'gw', options: gwOptions };
  }
  if (usesDmft) {
    const correlatedOrbitalIndices = document.getElementById('periodic-dmft-correlated-orbitals').value
      .split(/[\s,]+/)
      .map((item) => item.trim())
      .filter((item) => /^\d+$/.test(item))
      .map((item) => Number.parseInt(item, 10));
    const localizationMethod = document.getElementById('periodic-dmft-localization').value;
    payload.method = usesGw ? 'dft' : 'hf';
    payload.xc = usesGw ? (payload.xc || 'pbe') : null;
    payload.restricted = usesGw ? true : payload.restricted;
    payload.solver = {
      name: correlationTreatment,
      options: {
        ...gwOptions,
        impurity_solver: document.getElementById('periodic-dmft-impurity-solver').value,
        nbath: Math.max(2, integerInputValue('periodic-dmft-nbath', 4)),
        max_iterations: Math.max(1, integerInputValue('periodic-dmft-max-iterations', 10)),
        convergence_tolerance: numericInputValue('periodic-dmft-convergence', 1e-3),
      },
    };
    payload.embedding = {
      enabled: true,
      provider: 'fcdmft',
      localization_method: localizationMethod,
      minimal_basis: document.getElementById('periodic-dmft-minimal-basis').value.trim() || 'minao',
      include_pao: localizationMethod === 'iao_pao',
      correlated_orbital_indices: correlatedOrbitalIndices,
      approved: false,
    };
  }
  return payload;
}

function integerInputValue(id, defaultValue) {
  const parsed = Number.parseInt(document.getElementById(id).value, 10);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

function numericInputValue(id, defaultValue) {
  const parsed = Number.parseFloat(document.getElementById(id).value);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

function optionalNumericInputValue(id) {
  const value = document.getElementById(id).value.trim();
  if (!value) {
    return null;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function optionalIntegerTriplet(ids) {
  const values = ids.map((id) => document.getElementById(id).value.trim());
  if (values.every((value) => !value)) {
    return null;
  }
  return values.map((value) => Number.parseInt(value, 10));
}

function validatePeriodicFormControls() {
  const bandPathMode = document.getElementById('periodic-band-path-mode').value;
  const numericIds = [
    'periodic-kmesh-x', 'periodic-kmesh-y', 'periodic-kmesh-z',
    'periodic-kpoint-shift-x', 'periodic-kpoint-shift-y', 'periodic-kpoint-shift-z',
    'periodic-precision', 'periodic-ke-cutoff',
    'periodic-fft-mesh-x', 'periodic-fft-mesh-y', 'periodic-fft-mesh-z',
  ];
  if (bandPathMode === 'seekpath') {
    numericIds.push('periodic-band-path-reference-distance', 'periodic-band-path-symprec');
  } else {
    numericIds.push('periodic-band-path-npoints');
  }
  if (document.getElementById('periodic-smearing').value !== 'none') {
    numericIds.push('periodic-smearing-sigma');
  }
  const invalidInput = numericIds
    .map((id) => document.getElementById(id))
    .find((input) => input.value.trim() && !input.checkValidity());
  if (invalidInput) {
    return langText(
      'Review the periodic numerical controls; one or more values are outside the allowed range.',
      '请检查周期数值控制；至少一个数值超出了允许范围。'
    );
  }
  const fftValues = ['periodic-fft-mesh-x', 'periodic-fft-mesh-y', 'periodic-fft-mesh-z']
    .map((id) => document.getElementById(id).value.trim());
  const fftEntries = fftValues.filter(Boolean).length;
  if (fftEntries !== 0 && fftEntries !== 3) {
    return langText(
      'Provide all three FFT mesh dimensions, or leave all three blank for automatic selection.',
      'FFT 网格必须同时填写三个方向，或者全部留空并自动选择。'
    );
  }
  if (document.getElementById('periodic-ke-cutoff').value.trim() && fftEntries) {
    return langText(
      'Choose either a plane-wave auxiliary cutoff or an explicit FFT mesh, not both.',
      '平面波辅助截断和显式 FFT 网格只能选择一种。'
    );
  }
  const smearingMethod = document.getElementById('periodic-smearing').value;
  if (smearingMethod !== 'none' && !document.getElementById('periodic-smearing-sigma').value.trim()) {
    return langText(
      'Occupation smearing requires a positive sigma in Hartree.',
      '启用占据展宽时必须给出正的 sigma（Hartree）。'
    );
  }
  const bandStructureEnabled = Boolean(document.querySelector(
    '#periodic-output-group input[name="outputs"][value="band_structure"]:checked'
  ));
  if (bandStructureEnabled) {
    const mode = bandPathMode;
    const path = normalizePeriodicBandPath(document.getElementById('periodic-band-path').value);
    if (mode === 'custom' || mode === 'explicit') {
      if (!path) {
        return langText(
          'Provide a high-symmetry path for the selected band-path mode.',
          '当前能带路径模式需要填写高对称点路径。'
        );
      }
      const sections = path.split(',');
      const labels = [];
      for (const section of sections) {
        const sectionLabels = section.match(/[A-Z][a-z0-9]*/g) || [];
        if (sectionLabels.length < 2 || sectionLabels.join('') !== section) {
          return langText(
            'Use an ASE path such as GXWKGL or disconnected sections such as GX,LU.',
            '请使用 ASE 路径格式，例如 GXWKGL；不连续路径可写为 GX,LU。'
          );
        }
        labels.push(...sectionLabels);
      }
      if (mode === 'custom') {
        const preview = lastPeriodicStructurePreview && lastPeriodicStructurePreview.band_path_preview;
        const availablePoints = preview && preview.status === 'available'
          ? preview.special_points_scaled
          : null;
        if (availablePoints && typeof availablePoints === 'object') {
          const unknownLabels = Array.from(new Set(labels.filter((label) => !(label in availablePoints))));
          if (unknownLabels.length) {
            return langText(
              `Unknown standard special-point labels: ${unknownLabels.join(', ')}.`,
              `存在当前晶胞不支持的标准高对称点：${unknownLabels.join(', ')}。`
            );
          }
        }
      }
      if (mode === 'explicit') {
        const parsedPoints = parsePeriodicBandSpecialPoints(
          document.getElementById('periodic-band-special-points').value
        );
        if (parsedPoints.error) {
          return parsedPoints.error;
        }
        if (!Object.keys(parsedPoints.points).length) {
          return langText(
            'Explicit band paths require special-point coordinates.',
            '显式能带路径必须提供高对称点坐标。'
          );
        }
        const missingLabels = Array.from(new Set(labels.filter((label) => !(label in parsedPoints.points))));
        if (missingLabels.length) {
          return langText(
            `Missing explicit coordinates for: ${missingLabels.join(', ')}.`,
            `缺少以下高对称点的显式坐标：${missingLabels.join(', ')}。`
          );
        }
      }
    }
  }
  return '';
}
