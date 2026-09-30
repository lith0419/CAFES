const MOLECULAR_COVALENT_RADII = {
  H: 0.31, He: 0.28,
  Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Ne: 0.58,
  Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32,
  Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20,
  Br: 1.20, Kr: 1.16, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54,
  Tc: 1.47, Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39,
  Sb: 1.39, Te: 1.38, I: 1.39, Xe: 1.40,
};

const MOLECULAR_ELEMENT_COLORS = {
  H: '#ffffff', C: '#334155', N: '#2563eb', O: '#dc2626', F: '#16a34a',
  P: '#f97316', S: '#facc15', Cl: '#22c55e', Br: '#92400e', I: '#7c3aed',
  B: '#f5b7a7', Si: '#c084fc', Li: '#a78bfa', Na: '#60a5fa', K: '#818cf8',
  Mg: '#86efac', Ca: '#bef264', Fe: '#b45309', Cu: '#d97706', Zn: '#94a3b8',
};

function molecularPreviewNumber(value) {
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : null;
}

function molecularPreviewElementSymbol(rawSymbol) {
  const match = String(rawSymbol || '').trim().match(/^([A-Za-z]{1,2})/);
  if (!match) {
    return '';
  }
  const letters = match[1];
  return letters.charAt(0).toUpperCase() + letters.slice(1).toLowerCase();
}

function parseMolecularAtomString(atomText) {
  const atoms = [];
  String(atomText || '')
    .replace(/\r/g, '\n')
    .split(/[;\n]+/)
    .forEach((line, lineIndex) => {
      const cleaned = line.replace(/#.*/, '').trim();
      if (!cleaned) {
        return;
      }
      const tokens = cleaned.replace(/,/g, ' ').split(/\s+/).filter(Boolean);
      if (tokens.length < 4) {
        return;
      }
      const element = molecularPreviewElementSymbol(tokens[0]);
      const x = molecularPreviewNumber(tokens[1]);
      const y = molecularPreviewNumber(tokens[2]);
      const z = molecularPreviewNumber(tokens[3]);
      if (!element || x === null || y === null || z === null) {
        return;
      }
      atoms.push({ index: atoms.length, element, x, y, z, sourceLine: lineIndex + 1 });
    });
  return atoms;
}

function molecularPreviewDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function inferMolecularBonds(atoms) {
  const bonds = [];
  for (let i = 0; i < atoms.length; i += 1) {
    for (let j = i + 1; j < atoms.length; j += 1) {
      const a = atoms[i];
      const b = atoms[j];
      const distance = molecularPreviewDistance(a, b);
      const radiusA = MOLECULAR_COVALENT_RADII[a.element] || 0.77;
      const radiusB = MOLECULAR_COVALENT_RADII[b.element] || 0.77;
      const cutoff = Math.max(0.45, 1.25 * (radiusA + radiusB) + 0.15);
      if (distance > 0.1 && distance <= cutoff) {
        bonds.push({ source: i, target: j, distance });
      }
    }
  }
  return bonds;
}

function molecularPreviewEscape(value) {
  return AgentUI.escapeHtml(value);
}

function molecularPreviewFormatDistance(value) {
  if (!Number.isFinite(value)) {
    return '';
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
}

function molecularPreviewOptions(options) {
  return {
    unit: options && options.unit ? String(options.unit) : 'Angstrom',
    emptyText: options && options.emptyText ? String(options.emptyText) : 'No molecular structure specified.',
    parseErrorText: options && options.parseErrorText ? String(options.parseErrorText) : 'Could not parse molecular coordinates.',
    atomLabel: options && options.atomLabel ? String(options.atomLabel) : 'atoms',
    bondLabel: options && options.bondLabel ? String(options.bondLabel) : 'inferred bonds',
    unitLabel: options && options.unitLabel ? String(options.unitLabel) : 'unit',
    bondLengthLabel: options && options.bondLengthLabel ? String(options.bondLengthLabel) : 'bond length',
  };
}

function molecularPreviewProjector(atoms, width, height) {
  const yaw = -Math.PI / 6;
  const pitch = Math.PI / 9;
  const projected = atoms.map((atom) => {
    const x1 = atom.x * Math.cos(yaw) - atom.y * Math.sin(yaw);
    const y1 = atom.x * Math.sin(yaw) + atom.y * Math.cos(yaw);
    const y2 = y1 * Math.cos(pitch) - atom.z * Math.sin(pitch);
    const z2 = y1 * Math.sin(pitch) + atom.z * Math.cos(pitch);
    return { ...atom, px: x1, py: y2, depth: z2 };
  });
  const minX = Math.min(...projected.map((atom) => atom.px));
  const maxX = Math.max(...projected.map((atom) => atom.px));
  const minY = Math.min(...projected.map((atom) => atom.py));
  const maxY = Math.max(...projected.map((atom) => atom.py));
  const spanX = Math.max(maxX - minX, 0.8);
  const spanY = Math.max(maxY - minY, 0.8);
  const padding = 58;
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const xOffset = (width - spanX * scale) / 2;
  const yOffset = (height - spanY * scale) / 2;
  return projected.map((atom) => ({
    ...atom,
    sx: xOffset + (atom.px - minX) * scale,
    sy: height - (yOffset + (atom.py - minY) * scale),
  }));
}

function renderMolecularStructurePreview(targetOrId, atomText, options = {}) {
  const target = typeof targetOrId === 'string' ? document.getElementById(targetOrId) : targetOrId;
  if (!target) {
    return { status: 'missing_target', atoms: [], bonds: [] };
  }
  const labels = molecularPreviewOptions(options);
  if (!String(atomText || '').trim()) {
    target.innerHTML = `<div class="molecule-preview-empty">${molecularPreviewEscape(labels.emptyText)}</div>`;
    return { status: 'empty', atoms: [], bonds: [] };
  }
  const atoms = parseMolecularAtomString(atomText);
  if (!atoms.length) {
    target.innerHTML = `<div class="molecule-preview-empty">${molecularPreviewEscape(labels.parseErrorText)}</div>`;
    return { status: 'parse_failed', atoms: [], bonds: [] };
  }
  const bonds = inferMolecularBonds(atoms);
  const width = 720;
  const height = 360;
  const projected = molecularPreviewProjector(atoms, width, height);
  const atomMap = new Map(projected.map((atom) => [atom.index, atom]));
  const bondMarkup = bonds
    .map((bond) => {
      const a = atomMap.get(bond.source);
      const b = atomMap.get(bond.target);
      if (!a || !b) {
        return '';
      }
      const depth = (a.depth + b.depth) / 2;
      return { depth, markup: [
        `<line class="molecule-preview-bond" x1="${a.sx.toFixed(2)}" y1="${a.sy.toFixed(2)}" x2="${b.sx.toFixed(2)}" y2="${b.sy.toFixed(2)}" />`,
        `<line class="molecule-preview-bond-highlight" x1="${a.sx.toFixed(2)}" y1="${a.sy.toFixed(2)}" x2="${b.sx.toFixed(2)}" y2="${b.sy.toFixed(2)}" />`,
      ].join('') };
    })
    .sort((a, b) => a.depth - b.depth)
    .map((item) => item.markup)
    .join('');
  const minDepth = Math.min(...projected.map((atom) => atom.depth));
  const maxDepth = Math.max(...projected.map((atom) => atom.depth));
  const depthSpan = Math.max(maxDepth - minDepth, 1e-6);
  const atomMarkup = projected
    .slice()
    .sort((a, b) => a.depth - b.depth)
    .map((atom) => {
      const depthFactor = (atom.depth - minDepth) / depthSpan;
      const radius = 15 + depthFactor * 7;
      const color = MOLECULAR_ELEMENT_COLORS[atom.element] || '#cbd5e1';
      const stroke = atom.element === 'H' ? '#64748b' : '#0f172a';
      const labelColor = atom.element === 'H' || atom.element === 'S' ? '#0f172a' : '#ffffff';
      return [
        `<ellipse class="molecule-preview-depth-shadow" cx="${(atom.sx + 5).toFixed(2)}" cy="${(atom.sy + radius + 5).toFixed(2)}" rx="${(radius * 0.78).toFixed(2)}" ry="${(radius * 0.28).toFixed(2)}" />`,
        `<circle cx="${atom.sx.toFixed(2)}" cy="${atom.sy.toFixed(2)}" r="${radius.toFixed(2)}" fill="${color}" stroke="${stroke}" stroke-width="2.2" />`,
        `<circle cx="${(atom.sx - radius * 0.28).toFixed(2)}" cy="${(atom.sy - radius * 0.32).toFixed(2)}" r="${(radius * 0.22).toFixed(2)}" fill="rgba(255,255,255,0.55)" />`,
        `<text class="molecule-preview-atom-label" x="${atom.sx.toFixed(2)}" y="${atom.sy.toFixed(2)}" fill="${labelColor}">${molecularPreviewEscape(atom.element)}</text>`,
      ].join('');
    })
    .join('');
  const distancePills = bonds.slice(0, 4).map((bond) => {
    const a = atoms[bond.source];
    const b = atoms[bond.target];
    return `${a.element}-${b.element} ${molecularPreviewFormatDistance(bond.distance)} ${labels.unit}`;
  });
  const meta = [
    `${atoms.length} ${labels.atomLabel}`,
    `${bonds.length} ${labels.bondLabel}`,
    `${labels.unitLabel}=${labels.unit}`,
    ...distancePills.map((item) => `${labels.bondLengthLabel}: ${item}`),
  ];
  target.innerHTML = [
    `<div class="molecule-preview-meta">${meta.map((item) => `<span class="molecule-preview-pill">${molecularPreviewEscape(item)}</span>`).join('')}</div>`,
    '<div class="molecule-preview-canvas">',
    `<svg class="molecule-preview-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Molecular structure preview">`,
    '<g>',
    bondMarkup,
    atomMarkup,
    '</g>',
    '</svg>',
    '</div>',
  ].join('');
  return { status: 'rendered', atoms, bonds };
}
