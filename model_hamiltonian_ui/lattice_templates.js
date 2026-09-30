function translatedOffset(a, scaleA, b, scaleB) {
  return {
    x: scaleA * a.x + scaleB * b.x,
    y: scaleA * a.y + scaleB * b.y,
  };
}

function periodicOffsets(a, b) {
  return [
    translatedOffset(a, 1, b, 0),
    translatedOffset(a, -1, b, 0),
    translatedOffset(a, 0, b, 1),
    translatedOffset(a, 0, b, -1),
    translatedOffset(a, 1, b, 1),
    translatedOffset(a, -1, b, -1),
    translatedOffset(a, 1, b, -1),
    translatedOffset(a, -1, b, 1),
  ];
}

function isZeroOffset(offset) {
  return Math.abs(offset.x) <= 1e-12 && Math.abs(offset.y) <= 1e-12;
}

function uniqueOffsets(offsets) {
  const unique = [];
  offsets.forEach((offset) => {
    if (!unique.some((item) => Math.abs(item.x - offset.x) <= 1e-10 && Math.abs(item.y - offset.y) <= 1e-10)) {
      unique.push(offset);
    }
  });
  return unique;
}

function bondCandidateOffsets(vectors = {}, includePeriodic = state.boundary === 'periodic') {
  const offsets = [{ x: 0, y: 0 }];
  if (!includePeriodic || !vectors.a) {
    return offsets;
  }
  if (vectors.b) {
    offsets.push(...periodicOffsets(vectors.a, vectors.b));
  } else {
    offsets.push(vectors.a, { x: -vectors.a.x, y: -vectors.a.y });
  }
  return uniqueOffsets(offsets);
}

function addBondsByOffsetDistance(sites, bonds, vectors, nearestLength, tolerance = 1e-5, options = {}) {
  const maxDegree = Number.isFinite(options.maxDegree) ? options.maxDegree : Infinity;
  const pairFilter = typeof options.pairFilter === 'function' ? options.pairFilter : null;
  const indexHint = typeof options.indexHint === 'function' ? options.indexHint : null;
  const includePeriodic = options.periodic === undefined ? state.boundary === 'periodic' : Boolean(options.periodic);
  const candidates = bondCandidateOffsets(vectors, includePeriodic);
  const degree = new Map(sites.map((site) => [site.id, 0]));
  bonds.forEach((bond) => {
    degree.set(bond.source, (degree.get(bond.source) || 0) + 1);
    degree.set(bond.target, (degree.get(bond.target) || 0) + 1);
  });
  const orderedCandidates = [
    ...candidates.filter((offset) => isZeroOffset(offset)),
    ...candidates.filter((offset) => !isZeroOffset(offset)),
  ];
  orderedCandidates.forEach((offset) => {
    if (!isZeroOffset(offset)) {
      for (let i = 0; i < sites.length; i += 1) {
        const site = sites[i];
        if ((degree.get(site.id) || 0) + 2 > maxDegree) {
          continue;
        }
        if (pairFilter && !pairFilter(site, site)) {
          continue;
        }
        if (Math.abs(Math.hypot(offset.x, offset.y) - nearestLength) > tolerance) {
          continue;
        }
        const hint = indexHint ? indexHint(site, site, offset, true) : i;
        const previousBondCount = bonds.length;
        pushUniqueBond(bonds, site.id, site.id, hint, { periodic: true, offset });
        if (bonds.length > previousBondCount) {
          degree.set(site.id, (degree.get(site.id) || 0) + 2);
        }
      }
    }
    for (let i = 0; i < sites.length; i += 1) {
      for (let j = i + 1; j < sites.length; j += 1) {
        if ((degree.get(sites[i].id) || 0) >= maxDegree || (degree.get(sites[j].id) || 0) >= maxDegree) {
          continue;
        }
        if (pairFilter && !pairFilter(sites[i], sites[j])) {
          continue;
        }
        const dx = sites[j].x + offset.x - sites[i].x;
        const dy = sites[j].y + offset.y - sites[i].y;
        if (Math.abs(Math.hypot(dx, dy) - nearestLength) > tolerance) {
          continue;
        }
        const periodic = !isZeroOffset(offset);
        const hint = indexHint ? indexHint(sites[i], sites[j], offset, periodic) : i + j;
        const previousBondCount = bonds.length;
        pushUniqueBond(bonds, sites[i].id, sites[j].id, hint, { periodic, offset });
        if (bonds.length > previousBondCount) {
          degree.set(sites[i].id, (degree.get(sites[i].id) || 0) + 1);
          degree.set(sites[j].id, (degree.get(sites[j].id) || 0) + 1);
        }
      }
    }
  });
}

function renumberSitesSpatially(sites, bonds, yTolerance = 1e-5) {
  const ordered = sites.slice().sort((left, right) => {
    if (Math.abs(left.y - right.y) > yTolerance) {
      return left.y - right.y;
    }
    if (Math.abs(left.x - right.x) > yTolerance) {
      return left.x - right.x;
    }
    return left.id - right.id;
  });
  const remap = new Map();
  ordered.forEach((site, index) => {
    remap.set(site.id, index);
    site.id = index;
  });
  sites.sort((left, right) => left.id - right.id);
  bonds.forEach((bond) => {
    bond.source = remap.get(bond.source);
    bond.target = remap.get(bond.target);
  });
}

function cellWithVectors(a, b) {
  return {
    a: { x: a.x, y: a.y },
    b: { x: b.x, y: b.y },
    origin: { x: 0, y: 0 },
  };
}

function inverseCellCoordinates(point, a, b) {
  const determinant = a.x * b.y - a.y * b.x;
  if (Math.abs(determinant) < 1e-10) {
    return { u: 0, v: 0 };
  }
  return {
    u: (point.x * b.y - point.y * b.x) / determinant,
    v: (a.x * point.y - a.y * point.x) / determinant,
  };
}

function finalizeCellOrigin(cell, sites) {
  if (!cell || !sites.length) {
    return null;
  }
  const fractional = sites.map((site) => inverseCellCoordinates(site, cell.a, cell.b));
  const minU = Math.min(...fractional.map((point) => point.u));
  const maxU = Math.max(...fractional.map((point) => point.u));
  const minV = Math.min(...fractional.map((point) => point.v));
  const maxV = Math.max(...fractional.map((point) => point.v));
  const marginU = Math.max(0.08, (1 - (maxU - minU)) / 2);
  const marginV = Math.max(0.08, (1 - (maxV - minV)) / 2);
  return {
    ...cell,
    origin: {
      x: (minU - marginU) * cell.a.x + (minV - marginV) * cell.b.x,
      y: (minU - marginU) * cell.a.y + (minV - marginV) * cell.b.y,
    },
  };
}

function translateCell(cell, dx, dy) {
  if (!cell) {
    return null;
  }
  return {
    ...cell,
    origin: {
      x: cell.origin.x - dx,
      y: cell.origin.y - dy,
    },
  };
}

function createChain(size, ring = false, zigzag = false) {
  const sites = [];
  const bonds = [];
  const spacing = 1;
  const x0 = -((size - 1) * spacing) / 2;
  for (let i = 0; i < size; i += 1) {
    sites.push({
      id: i,
      x: x0 + i * spacing,
      y: zigzag ? (i % 2 === 0 ? 0.32 : -0.32) : 0,
      sublattice: i % 2 === 0 ? 'a' : 'b',
      epsilon: state.globals.epsilon,
      U: state.globals.U,
    });
  }
  const nearestLength = size > 1
    ? Math.hypot(sites[1].x - sites[0].x, sites[1].y - sites[0].y)
    : spacing;
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: size * spacing, y: 0 } },
    nearestLength,
    1e-5,
    {
      periodic: ring,
      indexHint: (left, right, offset, periodic) => (periodic ? size - 1 : Math.min(left.id, right.id)),
    },
  );
  return { sites, bonds };
}

function createRing(size) {
  const sites = [];
  const bonds = [];
  const radius = Math.max(1.4, size / (2 * Math.PI));
  for (let i = 0; i < size; i += 1) {
    const angle = (2 * Math.PI * i) / size - Math.PI / 2;
    sites.push({
      id: i,
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle),
      sublattice: i % 2 === 0 ? 'a' : 'b',
      epsilon: state.globals.epsilon,
      U: state.globals.U,
    });
  }
  for (let i = 0; i < size; i += 1) {
    pushUniqueBond(bonds, i, (i + 1) % size, i);
  }
  return { sites, bonds };
}

function createQuasi1D(length, width) {
  const sites = [];
  const bonds = [];
  const spacing = 1;
  const x0 = -((length - 1) * spacing) / 2;
  const y0 = -((width - 1) * 0.72) / 2;
  for (let rung = 0; rung < length; rung += 1) {
    for (let leg = 0; leg < width; leg += 1) {
      const id = rung * width + leg;
      sites.push({
        id,
        x: x0 + rung * spacing,
        y: y0 + leg * 0.72,
        sublattice: leg % 2 === 0 ? 'a' : 'b',
        epsilon: state.globals.epsilon,
        U: state.globals.U,
      });
    }
  }
  const sameLeg = (left, right) => left.id % width === right.id % width;
  const sameRungAdjacentLeg = (left, right) => (
    Math.floor(left.id / width) === Math.floor(right.id / width)
    && Math.abs((left.id % width) - (right.id % width)) === 1
  );
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: length * spacing, y: 0 } },
    spacing,
    1e-5,
    {
      pairFilter: sameLeg,
      indexHint: (left, right, offset, periodic) => (
        periodic
          ? length - 1
          : Math.min(Math.floor(left.id / width), Math.floor(right.id / width))
      ),
    },
  );
  addBondsByOffsetDistance(
    sites,
    bonds,
    {},
    0.72,
    1e-5,
    { periodic: false, pairFilter: sameRungAdjacentLeg, indexHint: () => 0 },
  );
  return { sites, bonds };
}

function primitiveCellMetadata(a, b, repetitions, basisSize) {
  return {
    vectors: [
      [a.x, a.y],
      [b.x, b.y],
    ],
    repetitions: repetitions.slice(),
    basis_size: basisSize,
  };
}

function addSiteToList(sites, x, y, sublattice = 'a', cellIndex = null, basisIndex = null) {
  const id = sites.length;
  const site = {
    id,
    x,
    y,
    sublattice,
    epsilon: state.globals.epsilon,
    U: state.globals.U,
  };
  if (Array.isArray(cellIndex)) {
    site.cellIndex = cellIndex.slice();
  }
  if (Number.isInteger(basisIndex)) {
    site.basisIndex = basisIndex;
  }
  sites.push(site);
  return id;
}

function createSquare(lx, ly) {
  const sites = [];
  const bonds = [];
  const spacing = 1;
  const x0 = -((lx - 1) * spacing) / 2;
  const y0 = -((ly - 1) * spacing) / 2;
  for (let y = 0; y < ly; y += 1) {
    for (let x = 0; x < lx; x += 1) {
      const id = y * lx + x;
      sites.push({
        id,
        x: x0 + x * spacing,
        y: y0 + y * spacing,
        sublattice: (x + y) % 2 === 0 ? 'a' : 'b',
        epsilon: state.globals.epsilon,
        U: state.globals.U,
        cellIndex: [x, y],
        basisIndex: 0,
      });
    }
  }
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: lx * spacing, y: 0 }, b: { x: 0, y: ly * spacing } },
    spacing,
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors({ x: lx * spacing, y: 0 }, { x: 0, y: ly * spacing }),
    primitiveCell: primitiveCellMetadata(
      { x: spacing, y: 0 },
      { x: 0, y: spacing },
      [lx, ly],
      1,
    ),
  };
}

function createLieb(lx, ly) {
  const sites = [];
  const bonds = [];
  const aVector = { x: lx, y: 0 };
  const bVector = { x: 0, y: ly };
  const nearestLength = 0.5;

  for (let y = 0; y < ly; y += 1) {
    for (let x = 0; x < lx; x += 1) {
      addSiteToList(sites, x, y, 'a', [x, y], 0);
      addSiteToList(sites, x + 0.5, y, 'b', [x, y], 1);
      addSiteToList(sites, x, y + 0.5, 'c', [x, y], 2);
    }
  }
  renumberSitesSpatially(sites, bonds, 1e-5);

  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: aVector, b: bVector },
    nearestLength,
    1e-5,
    { pairFilter: (left, right) => left.sublattice === 'a' || right.sublattice === 'a' },
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors(aVector, bVector),
    primitiveCell: primitiveCellMetadata(
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      [lx, ly],
      3,
    ),
  };
}

function createHoneycomb(lx, ly) {
  const sites = [];
  const bonds = [];
  const side = 0.72;
  const sqrt3 = Math.sqrt(3);
  const a1 = { x: sqrt3 * side, y: 0 };
  const a2 = { x: (sqrt3 * side) / 2, y: 1.5 * side };
  const basisB = { x: 0, y: side };

  for (let row = 0; row < ly; row += 1) {
    for (let col = 0; col < lx; col += 1) {
      const baseX = col * a1.x + row * a2.x;
      const baseY = col * a1.y + row * a2.y;
      addSiteToList(sites, baseX, baseY, 'a', [col, row], 0);
      addSiteToList(sites, baseX + basisB.x, baseY + basisB.y, 'b', [col, row], 1);
    }
  }
  renumberSitesSpatially(sites, bonds, 1e-5);
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: lx * a1.x, y: lx * a1.y }, b: { x: ly * a2.x, y: ly * a2.y } },
    side,
    1e-5,
    { pairFilter: (left, right) => left.sublattice !== right.sublattice },
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors({ x: lx * a1.x, y: lx * a1.y }, { x: ly * a2.x, y: ly * a2.y }),
    primitiveCell: primitiveCellMetadata(a1, a2, [lx, ly], 2),
  };
}

function createTriangular(lx, ly) {
  const sites = [];
  const bonds = [];
  const spacingY = Math.sqrt(3) / 2;
  for (let y = 0; y < ly; y += 1) {
    for (let x = 0; x < lx; x += 1) {
      addSiteToList(sites, x + 0.5 * y, spacingY * y, 'a', [x, y], 0);
    }
  }
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: lx, y: 0 }, b: { x: 0.5 * ly, y: spacingY * ly } },
    1,
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors({ x: lx, y: 0 }, { x: 0.5 * ly, y: spacingY * ly }),
    primitiveCell: primitiveCellMetadata(
      { x: 1, y: 0 },
      { x: 0.5, y: spacingY },
      [lx, ly],
      1,
    ),
  };
}

function createKagome(lx, ly) {
  const sites = [];
  const bonds = [];
  const sqrt3 = Math.sqrt(3);
  const a1 = { x: 2, y: 0 };
  const a2 = { x: 1, y: sqrt3 };
  const basis = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 0.5, y: sqrt3 / 2 },
  ];
  for (let y = 0; y < ly; y += 1) {
    for (let x = 0; x < lx; x += 1) {
      const baseX = x * a1.x + y * a2.x;
      const baseY = x * a1.y + y * a2.y;
      addSiteToList(sites, baseX + basis[0].x, baseY + basis[0].y, 'a', [x, y], 0);
      addSiteToList(sites, baseX + basis[1].x, baseY + basis[1].y, 'b', [x, y], 1);
      addSiteToList(sites, baseX + basis[2].x, baseY + basis[2].y, 'c', [x, y], 2);
    }
  }
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: lx * a1.x, y: lx * a1.y }, b: { x: ly * a2.x, y: ly * a2.y } },
    1,
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors({ x: lx * a1.x, y: lx * a1.y }, { x: ly * a2.x, y: ly * a2.y }),
    primitiveCell: primitiveCellMetadata(a1, a2, [lx, ly], 3),
  };
}

function createDice(lx, ly) {
  const sites = [];
  const bonds = [];
  const sqrt3 = Math.sqrt(3);
  const a1 = { x: 2, y: 0 };
  const a2 = { x: 1, y: sqrt3 };
  for (let y = 0; y < ly; y += 1) {
    for (let x = 0; x < lx; x += 1) {
      const baseX = x * a1.x + y * a2.x;
      const baseY = x * a1.y + y * a2.y;
      addSiteToList(sites, baseX, baseY, 'a', [x, y], 0);
      addSiteToList(sites, baseX - 1, baseY, 'b', [x, y], 1);
      addSiteToList(sites, baseX + 0.5, baseY + sqrt3 / 2, 'c', [x, y], 2);
      addSiteToList(sites, baseX - 0.5, baseY + sqrt3 / 2, 'd', [x, y], 3);
    }
  }
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: lx * a1.x, y: lx * a1.y }, b: { x: ly * a2.x, y: ly * a2.y } },
    1,
    1e-5,
    { pairFilter: (left, right) => left.sublattice === 'a' || right.sublattice === 'a' },
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors({ x: lx * a1.x, y: lx * a1.y }, { x: ly * a2.x, y: ly * a2.y }),
    primitiveCell: primitiveCellMetadata(a1, a2, [lx, ly], 4),
  };
}

function createSquareOctagon(lx, ly) {
  const sites = [];
  const bonds = [];
  const edge = 1;
  const r = edge / Math.sqrt(2);
  const spacing = edge + 2 * r;
  for (let y = 0; y < ly; y += 1) {
    for (let x = 0; x < lx; x += 1) {
      const centerX = x * spacing;
      const centerY = y * spacing;
      addSiteToList(sites, centerX, centerY - r, 'a', [x, y], 0);
      addSiteToList(sites, centerX + r, centerY, 'b', [x, y], 1);
      addSiteToList(sites, centerX, centerY + r, 'c', [x, y], 2);
      addSiteToList(sites, centerX - r, centerY, 'd', [x, y], 3);
    }
  }
  addBondsByOffsetDistance(
    sites,
    bonds,
    { a: { x: lx * spacing, y: 0 }, b: { x: 0, y: ly * spacing } },
    edge,
  );
  return {
    sites,
    bonds,
    cell: cellWithVectors({ x: lx * spacing, y: 0 }, { x: 0, y: ly * spacing }),
    primitiveCell: primitiveCellMetadata(
      { x: spacing, y: 0 },
      { x: 0, y: spacing },
      [lx, ly],
      4,
    ),
  };
}

function compareNumberArrays(left, right) {
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const difference = (left[index] || 0) - (right[index] || 0);
    if (Math.abs(difference) > 1e-12) {
      return difference;
    }
  }
  return 0;
}

function blochTranslationVectors(dimension) {
  if (dimension === 1) {
    return [[-1], [0], [1]];
  }
  const translations = [];
  for (let x = -1; x <= 1; x += 1) {
    for (let y = -1; y <= 1; y += 1) {
      translations.push([x, y]);
    }
  }
  return translations;
}

function physicalBlochOffset(cellOffset, latticeVectors) {
  return cellOffset.reduce((offset, scale, index) => ({
    x: offset.x + scale * latticeVectors[index].x,
    y: offset.y + scale * latticeVectors[index].y,
  }), { x: 0, y: 0 });
}

function pushBlochBond(bonds, source, target, cellOffset, latticeVectors, kind = 'custom') {
  const reverse = [target, source, ...cellOffset.map((value) => -value)];
  const forward = [source, target, ...cellOffset];
  const canonical = compareNumberArrays(forward, reverse) <= 0 ? forward : reverse;
  const key = canonical.join(':');
  if (bonds.some((bond) => bond.blochKey === key)) {
    return;
  }
  const offset = physicalBlochOffset(cellOffset, latticeVectors);
  bonds.push({
    id: bonds.length,
    source,
    target,
    t: baseHoppingForKind(kind),
    V: state.globals.V,
    kind,
    periodic: cellOffset.some((value) => value !== 0),
    offset,
    cellOffset: cellOffset.slice(),
    addHermitianConjugate: true,
    blochKey: key,
  });
}

function addBlochBondsByDistance(sites, bonds, latticeVectors, nearestLength, tolerance = 1e-5, pairFilter = null) {
  const translations = blochTranslationVectors(latticeVectors.length);
  for (let sourceIndex = 0; sourceIndex < sites.length; sourceIndex += 1) {
    for (let targetIndex = 0; targetIndex < sites.length; targetIndex += 1) {
      const source = sites[sourceIndex];
      const target = sites[targetIndex];
      if (pairFilter && !pairFilter(source, target)) {
        continue;
      }
      translations.forEach((cellOffset) => {
        if (source.id === target.id && cellOffset.every((value) => value === 0)) {
          return;
        }
        const offset = physicalBlochOffset(cellOffset, latticeVectors);
        const distance = Math.hypot(target.x + offset.x - source.x, target.y + offset.y - source.y);
        if (Math.abs(distance - nearestLength) > tolerance) {
          return;
        }
        const forward = [source.id, target.id, ...cellOffset];
        const reverse = [target.id, source.id, ...cellOffset.map((value) => -value)];
        if (compareNumberArrays(forward, reverse) > 0) {
          return;
        }
        const periodic = cellOffset.some((value) => value !== 0);
        const kind = isSshBondModulationEnabled() ? (periodic ? 'weak' : 'strong') : 'custom';
        pushBlochBond(bonds, source.id, target.id, cellOffset, latticeVectors, kind);
      });
    }
  }
}

function blochCell(latticeVectors) {
  const a = latticeVectors[0];
  const b = latticeVectors[1] || { x: -a.y, y: a.x || 1 };
  return cellWithVectors(a, b);
}

function createBlochPrimitive(preset, width = 2) {
  const sites = [];
  const bonds = [];
  let latticeVectors;
  let nearestLength;
  let pairFilter = null;

  if (preset === 'quasi_1d') {
    latticeVectors = [{ x: 1, y: 0 }];
    const y0 = -((width - 1) * 0.72) / 2;
    for (let leg = 0; leg < width; leg += 1) {
      addSiteToList(sites, 0, y0 + leg * 0.72, leg % 2 === 0 ? 'a' : 'b');
      pushBlochBond(bonds, leg, leg, [1], latticeVectors, 'custom');
      if (leg > 0) {
        pushBlochBond(bonds, leg - 1, leg, [0], latticeVectors, 'custom');
      }
    }
    return { sites, bonds, latticeVectors, cell: blochCell(latticeVectors) };
  }

  if (preset === 'chain' || preset === 'ring') {
    if (isSshBondModulationEnabled()) {
      latticeVectors = [{ x: 2, y: 0 }];
      addSiteToList(sites, 0, 0, 'a');
      addSiteToList(sites, 1, 0, 'b');
      nearestLength = 1;
    } else {
      latticeVectors = [{ x: 1, y: 0 }];
      addSiteToList(sites, 0, 0, 'a');
      nearestLength = 1;
    }
  } else if (preset === 'zigzag') {
    latticeVectors = [{ x: 2, y: 0 }];
    addSiteToList(sites, 0, 0.32, 'a');
    addSiteToList(sites, 1, -0.32, 'b');
    nearestLength = Math.hypot(1, 0.64);
  } else if (preset === 'square') {
    latticeVectors = [{ x: 1, y: 0 }, { x: 0, y: 1 }];
    addSiteToList(sites, 0, 0, 'a');
    nearestLength = 1;
  } else if (preset === 'lieb') {
    latticeVectors = [{ x: 1, y: 0 }, { x: 0, y: 1 }];
    addSiteToList(sites, 0, 0, 'a');
    addSiteToList(sites, 0.5, 0, 'b');
    addSiteToList(sites, 0, 0.5, 'c');
    nearestLength = 0.5;
    pairFilter = (left, right) => left.sublattice === 'a' || right.sublattice === 'a';
  } else if (preset === 'honeycomb') {
    const side = 0.72;
    latticeVectors = [
      { x: Math.sqrt(3) * side, y: 0 },
      { x: (Math.sqrt(3) * side) / 2, y: 1.5 * side },
    ];
    addSiteToList(sites, 0, 0, 'a');
    addSiteToList(sites, 0, side, 'b');
    nearestLength = side;
    pairFilter = (left, right) => left.sublattice !== right.sublattice;
  } else if (preset === 'triangular') {
    latticeVectors = [{ x: 1, y: 0 }, { x: 0.5, y: Math.sqrt(3) / 2 }];
    addSiteToList(sites, 0, 0, 'a');
    nearestLength = 1;
  } else if (preset === 'kagome') {
    latticeVectors = [{ x: 2, y: 0 }, { x: 1, y: Math.sqrt(3) }];
    addSiteToList(sites, 0, 0, 'a');
    addSiteToList(sites, 1, 0, 'b');
    addSiteToList(sites, 0.5, Math.sqrt(3) / 2, 'c');
    nearestLength = 1;
  } else if (preset === 'dice') {
    latticeVectors = [{ x: 2, y: 0 }, { x: 1, y: Math.sqrt(3) }];
    addSiteToList(sites, 0, 0, 'a');
    addSiteToList(sites, -1, 0, 'b');
    addSiteToList(sites, 0.5, Math.sqrt(3) / 2, 'c');
    addSiteToList(sites, -0.5, Math.sqrt(3) / 2, 'b');
    nearestLength = 1;
    pairFilter = (left, right) => left.sublattice === 'a' || right.sublattice === 'a';
  } else {
    const edge = 1;
    const radius = edge / Math.sqrt(2);
    const spacing = edge + 2 * radius;
    latticeVectors = [{ x: spacing, y: 0 }, { x: 0, y: spacing }];
    addSiteToList(sites, 0, -radius, 'a');
    addSiteToList(sites, radius, 0, 'b');
    addSiteToList(sites, 0, radius, 'c');
    addSiteToList(sites, -radius, 0, 'd');
    nearestLength = edge;
  }

  addBlochBondsByDistance(sites, bonds, latticeVectors, nearestLength, 1e-5, pairFilter);
  return { sites, bonds, latticeVectors, cell: blochCell(latticeVectors) };
}

function buildLattice() {
  updateGlobalsFromControls();
  const minimumSizeX = !isBlochRepresentation() && state.dimension === 2 ? 1 : 2;
  const sizeX = clampInt(numberValue('size-x', 8), minimumSizeX, 64);
  let sizeY = clampInt(numberValue('size-y', 1), 1, 16);
  document.getElementById('size-x').value = String(sizeX);
  let lattice;
  if (isBlochRepresentation()) {
    state.preset = state.dimension === 1 ? state.template1d : state.template2d;
    if (state.preset === 'ring') {
      state.preset = 'chain';
      state.template1d = 'chain';
      document.getElementById('template-1d').value = 'chain';
    }
    sizeY = state.preset === 'quasi_1d' ? clampInt(sizeY, 2, 16) : 1;
    lattice = createBlochPrimitive(state.preset, sizeY);
  } else if (state.dimension === 1) {
    state.dimension = 1;
    state.preset = state.template1d;
    if (state.template1d === 'quasi_1d') {
      sizeY = clampInt(sizeY, 2, 16);
      document.getElementById('size-y').value = String(sizeY);
      lattice = createQuasi1D(sizeX, sizeY);
    } else if (state.template1d === 'ring') {
      sizeY = 1;
      document.getElementById('size-y').value = String(sizeY);
      lattice = createRing(sizeX);
    } else {
      sizeY = 1;
      document.getElementById('size-y').value = String(sizeY);
      lattice = createChain(sizeX, state.boundary === 'periodic', state.template1d === 'zigzag');
    }
  } else {
    state.dimension = 2;
    state.preset = state.template2d;
    sizeY = clampInt(sizeY, 1, 16);
    document.getElementById('size-y').value = String(sizeY);
    if (state.template2d === 'square') {
      lattice = createSquare(sizeX, sizeY);
    } else if (state.template2d === 'lieb') {
      lattice = createLieb(sizeX, sizeY);
    } else if (state.template2d === 'honeycomb') {
      lattice = createHoneycomb(sizeX, sizeY);
    } else if (state.template2d === 'triangular') {
      lattice = createTriangular(sizeX, sizeY);
    } else if (state.template2d === 'kagome') {
      lattice = createKagome(sizeX, sizeY);
    } else if (state.template2d === 'dice') {
      lattice = createDice(sizeX, sizeY);
    } else {
      lattice = createSquareOctagon(sizeX, sizeY);
    }
  }
  state.sites = lattice.sites.map((site) => applyDefaultSiteParameters(site));
  state.bonds = lattice.bonds;
  state.cell = lattice.cell ? finalizeCellOrigin(lattice.cell, lattice.sites) : null;
  state.primitiveCell = lattice.primitiveCell
    ? JSON.parse(JSON.stringify(lattice.primitiveCell))
    : null;
  state.cellSubdivisions = state.dimension === 2 && !isBlochRepresentation()
    ? { x: sizeX, y: sizeY }
    : { x: 1, y: 1 };
  state.latticeVectors = (lattice.latticeVectors || []).map((vector) => ({ ...vector }));
  state.selected = null;
  state.pendingBondSource = null;
  if (isBlochRepresentation()) {
    state.boundary = 'periodic';
    state.occupation.electronsPerCell = state.sites.length;
  } else {
    state.nelec = [Math.ceil(state.sites.length / 2), Math.floor(state.sites.length / 2)];
  }
  updateControlsFromState();
  centerLayout();
  render();
}

function centerLayout() {
  if (!state.sites.length) {
    return;
  }
  const minX = Math.min(...state.sites.map((site) => site.x));
  const maxX = Math.max(...state.sites.map((site) => site.x));
  const minY = Math.min(...state.sites.map((site) => site.y));
  const maxY = Math.max(...state.sites.map((site) => site.y));
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  state.sites.forEach((site) => {
    site.x -= centerX;
    site.y -= centerY;
  });
  state.cell = translateCell(state.cell, centerX, centerY);
}
