#!/usr/bin/env python3
"""Select 500 distinct molecular graphs and their original QM9 geometries.

No quantum calculations are executed. Dependencies: numpy, rdkit, pyscf, Pillow.
python select_seeds.py --source-dir /path/to/qm9-source
Optional --cache-dir caches the screened source catalog for repeat runs.
"""
from __future__ import annotations

from tools.datasets.qm9.common import digest as sha, numeric, canonical, OFFICIAL_SOURCE_FILES as FILES, DIFFICULT_QM9_IDS as DIFFICULT

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyscf
import rdkit
from pyscf import gto
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDetermineBonds, rdFingerprintGenerator, rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers

SEED = 20260921
QUOTAS = {'1-3': 15, '4-5': 65, '6': 130, '7': 190, '8': 70, '9': 30}
SMARTS = {
    'nitrile': '[CX2]#[NX1]', 'alkyne': '[CX2]#[CX2]',
    'alkene': '[CX3]=[CX3]', 'alcohol_or_phenol': '[OX2H][#6]',
    'ether': '[OD2]([#6])[#6]', 'carbonyl': '[CX3]=[OX1]',
    'aldehyde': '[CX3H1](=O)[#6]', 'ketone': '[#6][CX3](=O)[#6]',
    'amide': '[NX3][CX3](=[OX1])', 'ester': '[CX3](=O)[OX2][#6]',
    'amine_non_amide': '[NX3;!$(N-C=O);!$(N=*)]',
    'aromatic_nitrogen': '[nH0,nH1]', 'fluorinated': '[F]',
}




def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')




def bucket(n):
    return '1-3' if n <= 3 else '4-5' if n <= 5 else str(n)




def parse_record(raw, member):
    lines = raw.splitlines()
    n = int(lines[0])
    qid = int(lines[1].split()[1])
    symbols = [s.split()[0] for s in lines[2:2+n]]
    positions = [[numeric(v) for v in s.split()[1:4]] for s in lines[2:2+n]]
    smiles = lines[n+3].split()
    frequencies = [numeric(v) for v in lines[n+2].split()]
    return dict(qm9_id=qid, member=member, raw=raw, n_atoms=n,
                symbols=symbols, positions=positions,
                smiles_gdb9=smiles[0], smiles_optimized=smiles[1],
                min_source_frequency_cm1=min(frequencies) if frequencies else None)


def screen_sources(source, excluded, cache):
    if cache and (cache / 'catalog.jsonl').exists() and (cache / 'screening.json').exists():
        info = json.loads((cache / 'screening.json').read_text())
        if info.get('policy_version') == 'v2' and info.get('rdkit_version') == rdkit.__version__:
            print('Reading cached screened source catalog', flush=True)
            return [json.loads(s) for s in (cache / 'catalog.jsonl').read_text().splitlines()], info
    rejected = Counter()
    accepted = []
    seen = set()
    scanned = 0
    # These structural exclusions keep this first campaign away from peroxides,
    # single N-N bonds, highly polycyclic graphs and highly N/O-rich systems.
    # They do not establish single-reference suitability.
    with tarfile.open(source / 'dsgdb9nsd.xyz.tar.bz2', 'r|bz2') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.xyz'):
                continue
            raw = archive.extractfile(member).read().decode()
            item = parse_record(raw, member.name)
            scanned += 1
            if scanned % 20000 == 0:
                print(f'Screened {scanned}/133885, eligible {len(accepted)}', flush=True)
            qid = item['qm9_id']
            if qid in excluded:
                rejected['official_geometry_inconsistency'] += 1
                continue
            if qid in DIFFICULT:
                rejected['official_difficult_geometry'] += 1
                continue
            counts = Counter(item['symbols'])
            if counts['N'] > 3 or counts['O'] > 3 or counts['F'] > 4:
                rejected['element_count_policy'] += 1
                continue
            if item['min_source_frequency_cm1'] is None or item['min_source_frequency_cm1'] <= 0:
                rejected['nonpositive_or_missing_source_frequency'] += 1
                continue
            mol = Chem.MolFromSmiles(item['smiles_gdb9'])
            optimized = Chem.MolFromSmiles(item['smiles_optimized'])
            if mol is None or optimized is None:
                rejected['smiles_parse'] += 1
                continue
            # GDB input SMILES often omit stereochemistry that optimized SMILES
            # specify. Compare connectivity here; validate the optimized stereo
            # against XYZ later, without discarding otherwise valid chiral seeds.
            if canonical(mol, stereo=False) != canonical(optimized, stereo=False):
                rejected['original_optimized_connectivity_mismatch'] += 1
                continue
            if len(Chem.GetMolFrags(mol)) != 1:
                rejected['disconnected_graph'] += 1
                continue
            if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                rejected['formal_charge_or_radical'] += 1
                continue
            if mol.GetRingInfo().NumRings() > 2:
                rejected['more_than_two_rings'] += 1
                continue
            if any(b.GetBondType() == Chem.BondType.SINGLE and
                   b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() and
                   b.GetBeginAtom().GetAtomicNum() in (7, 8) for b in mol.GetBonds()):
                rejected['single_NN_or_OO_bond'] += 1
                continue
            if Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()) != counts:
                rejected['xyz_smiles_formula_mismatch'] += 1
                continue
            graph = canonical(mol, stereo=False)
            if graph in seen:
                rejected['duplicate_constitutional_graph'] += 1
                continue
            seen.add(graph)
            item.update(smiles=canonical(optimized), graph_smiles=graph,
                        formula=rdMolDescriptors.CalcMolFormula(mol),
                        n_heavy_atoms=mol.GetNumHeavyAtoms(),
                        ring_count=mol.GetRingInfo().NumRings(),
                        rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol))
            accepted.append(item)
    assert scanned == 133885
    accepted.sort(key=lambda r: r['qm9_id'])
    info = dict(policy_version='v2', rdkit_version=rdkit.__version__,
                records_scanned=scanned, accepted_candidates=len(accepted),
                rejection_counts=dict(rejected),
                candidate_strata=dict(Counter(bucket(r['n_heavy_atoms']) for r in accepted)))
    assert len(accepted) + sum(rejected.values()) == scanned
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / 'catalog.jsonl').open('w') as f:
            for r in accepted:
                f.write(json.dumps(r) + '\n')
        dump(cache / 'screening.json', info)
    return accepted, info


def validate_xyz(item):
    positions = np.array(item['positions'])
    if not np.isfinite(positions).all():
        return None, 'nonfinite_coordinates'
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    mindist = float(distances[np.triu_indices(len(positions), 1)].min())
    if mindist <= 0.5:
        return None, 'minimum_distance_le_0.5_angstrom'
    n = item['n_atoms']
    lines = item['raw'].splitlines()
    body = '\n'.join(' '.join(s.split()[:4]).replace('*^', 'e') for s in lines[2:2+n])
    xyz = f'{n}\nqm9_{item["qm9_id"]:06d} | charge=0 spin=0 | Angstrom | original QM9 geometry\n{body}\n'
    try:
        mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(mol, charge=0, allowChargedFragments=True)
        if canonical(mol) != item['smiles']:
            return None, 'xyz_graph_or_stereochemistry_mismatch'
    except Exception as exc:
        return None, 'xyz_bond_assignment_' + type(exc).__name__
    numbers = [Chem.GetPeriodicTable().GetAtomicNumber(x) for x in item['symbols']]
    if sum(numbers) % 2:
        return None, 'odd_electron_count'
    pmol = gto.M(atom=list(zip(item['symbols'], item['positions'])), basis='def2-svp',
                 unit='Angstrom', charge=0, spin=0, verbose=0)
    assert pmol.nelectron == sum(numbers)
    return dict(xyz=xyz, atomic_numbers=numbers, n_electrons=sum(numbers),
                n_ao_def2_svp=pmol.nao_nr(), min_distance_angstrom=mindist), None


def select(candidates, pinned):
    by_id = {r['qm9_id']: r for r in candidates}
    missing = set(pinned) - set(by_id)
    if missing:
        raise ValueError('Pinned pilot records fail screening: ' + str(sorted(missing)))
    selected, failures = [], []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                          includeChirality=False)
    for group, quota in QUOTAS.items():
        pool = [r for r in candidates if bucket(r['n_heavy_atoms']) == group]
        first = [i for i, r in enumerate(pool) if r['qm9_id'] in pinned]
        fps = [generator.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in pool]
        if len(pool) < quota:
            raise ValueError(f'Not enough eligible candidates in {group}: {len(pool)} < {quota}')
        ranked = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(
            fps, len(pool), min(len(pool), quota+150), firstPicks=first, seed=SEED)
        accepted = 0
        for idx in ranked:
            item = pool[idx]
            validation, reason = validate_xyz(item)
            if reason:
                failures.append(dict(qm9_id=item['qm9_id'], stratum=group, reason=reason))
                if item['qm9_id'] in pinned:
                    raise ValueError(f'Pinned pilot molecule {item["qm9_id"]}: {reason}')
                continue
            item.update(validation)
            item['stratum'] = group
            item['stratum_pick_rank'] = accepted + 1
            selected.append(item)
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise ValueError(f'Geometry-qualified count in {group}: {accepted} != {quota}')
        print(f'Selected stratum {group}: {accepted}, from {len(pool)} candidates', flush=True)
    selected.sort(key=lambda r: (r['n_heavy_atoms'], r['qm9_id']))
    assert len(selected) == 500
    assert len({r['graph_smiles'] for r in selected}) == 500
    assert set(pinned).issubset(r['qm9_id'] for r in selected)
    return selected, failures


def write_results(out, selected, pinned, sources, screening, failures):
    for name in ['xyz', 'source_originals', 'provenance', 'previews']:
        (out / name).mkdir(parents=True, exist_ok=True)
    # Rebuild only these generated artifacts so reruns cannot leave extra seeds.
    for name, pattern in [('xyz', 'qm9_*.xyz'), ('source_originals', 'dsgdb9nsd_*.xyz'),
                          ('previews', 'molecules_*.png')]:
        for old in (out / name).glob(pattern):
            old.unlink()
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}
    rows, geometries = [], []
    for order, item in enumerate(selected, 1):
        qid = item['qm9_id']
        mid = f'qm9_{qid:06d}'
        mol = Chem.MolFromSmiles(item['smiles'])
        tags = [name for name, pattern in patterns.items() if mol.HasSubstructMatch(pattern)]
        if any(a.GetIsAromatic() for a in mol.GetAtoms()):
            tags.append('aromatic')
        if item['ring_count']:
            tags.append('cyclic')
        if all(s in ('C', 'H') for s in item['symbols']):
            tags.append('hydrocarbon')
        pilot = pinned.get(qid, {})
        row = dict(selection_order=order, molecule_id=mid, qm9_id=qid,
                   name=pilot.get('name', ''), name_zh=pilot.get('name_zh', ''),
                   included_in_pilot40=bool(pilot), formula=item['formula'],
                   smiles=item['smiles'], smiles_gdb9=item['smiles_gdb9'],
                   smiles_optimized=item['smiles_optimized'],
                   constitutional_graph_smiles=item['graph_smiles'],
                   n_atoms=item['n_atoms'], n_heavy_atoms=item['n_heavy_atoms'],
                   n_electrons=item['n_electrons'], n_ao_def2_svp=item['n_ao_def2_svp'],
                   stratum=item['stratum'], stratum_pick_rank=item['stratum_pick_rank'],
                   ring_count=item['ring_count'], rotatable_bonds=item['rotatable_bonds'],
                   functional_tags=';'.join(tags), elements=';'.join(sorted(set(item['symbols']))),
                   charge=0, spin=0, coordinate_unit='Angstrom',
                   min_distance_angstrom=item['min_distance_angstrom'],
                   min_source_frequency_cm1=item['min_source_frequency_cm1'],
                   xyz_path=f'xyz/{mid}.xyz', source_member=item['member'],
                   source_sha256=hashlib.sha256(item['raw'].encode()).hexdigest(),
                   geometry_level='B3LYP/6-31G(2df,p)', ccsdt_suitability='not_assessed')
        rows.append(row)
        (out / row['xyz_path']).write_text(item['xyz'])
        (out / 'source_originals' / Path(item['member']).name).write_text(item['raw'])
        geometries.append(dict(molecule_id=mid, geometry_id='qm9_optimized_seed',
                               atomic_numbers=item['atomic_numbers'], positions=item['positions'],
                               coordinate_unit='Angstrom', charge=0, spin=0,
                               source={'dataset': 'QM9', 'qm9_id': qid,
                                       'geometry_level': row['geometry_level'],
                                       'archive_member': item['member'], 'sha256': row['source_sha256'],
                                       'doi': '10.6084/m9.figshare.978904_D12'}))
    dump(out / 'selection.json', rows)
    dump(out / 'seed_geometries.json', {'seed_geometries': geometries})
    with (out / 'selection.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'seeds.xyz').write_text(''.join(r['xyz'] for r in selected))
    (out / 'molecule_ids.txt').write_text(''.join(r['molecule_id'] + '\n' for r in rows))
    dump(out / 'geometry_rejections.json', failures)
    for start in range(0, 500, 50):
        page = rows[start:start+50]
        mols = [Chem.MolFromSmiles(r['smiles']) for r in page]
        legends = [f"{r['selection_order']:03d} | QM9 {r['qm9_id']}\n{r['formula']} | {r['n_heavy_atoms']} heavy | {r['n_ao_def2_svp']} AO" for r in page]
        image = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(320, 230),
                                    legends=legends, legendFontSize=17, legendFraction=0.22)
        image.save(out / 'previews' / f'molecules_{start//50+1:02d}.png')
    manifest = dict(schema='qm9-seeds-500-v1', molecule_count=500, structure_count=500,
                    structures_per_molecule=1, included_pilot_molecules=40,
                    source_files=sources, source_license='CC0 (Figshare metadata)',
                    source_geometry_level='B3LYP/6-31G(2df,p)', coordinate_unit='Angstrom',
                    coordinates_modified=False, new_quantum_calculations=0,
                    selection={'stratum_quotas': QUOTAS, 'seed': SEED,
                               'method': 'RDKit MaxMin within each heavy-atom stratum; pilot40 pinned first.',
                               'fingerprint': 'Morgan radius=2, 2048 bits, no chirality; Tanimoto distance',
                               'uniqueness': 'One geometry per stereo-insensitive canonical molecular graph',
                               'candidate_deduplication': 'First encountered eligible record in original archive',
                               'is_population_representative': False,
                               'screening_policy': ['Exclude official 3054 geometry-inconsistent records',
                                                    'Exclude 11 difficult geometries named in source README',
                                                    'Require positive published harmonic frequencies',
                                                    'Require original and optimized stereo-insensitive connectivity agreement; preserve optimized stereo',
                                                    'Connected graph; no atom formal charges or radicals',
                                                    'At most 3 N, 3 O, 4 F, and 2 RDKit SSSR rings',
                                                    'Exclude single N-N and O-O bonds',
                                                    'Require formula agreement between source XYZ and SMILES'],
                               'validation_fallback': 'Continue down same MaxMin ranking on rejected geometry; never relax filters'},
                    screening=screening, geometry_selection_rejections=len(failures),
                    heavy_atom_counts=dict(sorted(Counter(r['n_heavy_atoms'] for r in rows).items())),
                    ao_def2_svp_range=[min(r['n_ao_def2_svp'] for r in rows), max(r['n_ao_def2_svp'] for r in rows)],
                    element_molecule_counts={s: sum(s in r['elements'].split(';') for r in rows) for s in ['H','C','N','O','F']},
                    functional_tag_counts=dict(Counter(t for r in rows for t in r['functional_tags'].split(';') if t)),
                    rdkit_version=rdkit.__version__, pyscf_version=pyscf.__version__,
                    validation={'source_md5': 'passed', '500_unique_molecular_graphs': 'passed',
                                'pilot40_preserved': 'passed', 'finite_coordinates_and_min_distance': 'passed',
                                'xyz_graph_and_stereochemistry_vs_source_smiles': 'passed',
                                'even_electron_count': 'passed', 'positive_source_frequencies': 'passed',
                                'pyscf_def2_svp_molecule_build': 'passed'},
                    ccsdt_suitability='Not assessed. No SCF, stability analysis, CC calculation or new geometry optimization.')
    dump(out / 'manifest.json', manifest)
    table = ['|重原子数|分子/结构数量|', '|---|---:|'] + [f'|{k}|{v}|' for k,v in QUOTAS.items()]
    overview = '\n'.join(f'- [结构图 {i:02d}：{(i-1)*50+1}–{i*50}](previews/molecules_{i:02d}.png)' for i in range(1,11))
    readme = '''# 500 个 QM9 初始结构

共 **500 个不同分子，每个 1 个初始结构**，包含之前 40 分子候选集的全部原始坐标。
所有分子来自 QM9，保留 B3LYP/6-31G(2df,p) 优化坐标、原子顺序和原始方向；坐标单位 Å。
分子按不含立体化学的连接图去重，因此没有把同一连接图的多个立体异构体重复计为不同分子。

## 文件

- `xyz/`：500 个独立、含全部氢原子的标准 XYZ。
- `seeds.xyz`：合并多帧 XYZ；每帧是一个不同分子。
- `seed_geometries.json`：agent 的 `MolecularGeometry` 种子数组，电荷 0，spin=0。
- `selection.csv` / `selection.json`：QM9 ID、SMILES、分子式、重原子数、AO 数、结构标签。
- `molecule_ids.txt`：500 个稳定分子 ID。
- `source_originals/`：500 份原始 QM9 扩展 XYZ。
- `manifest.json`：筛选方法、各项计数、验证结果和来源校验值。
- `geometry_rejections.json`：指纹选择后被三维几何复核拒绝的候选和原因。
- `provenance/`：官方说明、排除清单、Figshare 元数据、已保留的原始 40 个分子清单。
- `SHA256SUMS`：交付文件校验值。

## 分层选择

''' + '\n'.join(table) + '''

400 个分子不超过 7 个重原子；100 个为 8–9 重原子。
每个尺寸档保留原始 pilot40 条目，然后按 Morgan 指纹（半径 2、2048 位、不编码手性）
的 Tanimoto 距离用 RDKit MaxMin 挑选，固定 seed=20260921。
这是一份兼顾成本与结构覆盖的设计子集，不代表 QM9 的天然分布，也不是经能量诊断筛选的弱相关集。

初步范围限定为中性、无分子图自由基、单连通分子；每分子 N≤3、O≤3、F≤4，
RDKit SSSR 环数≤2，排除单键 N–N/O–O，避免首批大量落在过氧化物、高杂原子密度或多笼状体系。
这些约束是本次试验设计，不表示被排除类别不能计算 CCSD(T)。
官能团标签为 SMARTS 匹配，允许重叠，不是互斥分类。

## 验证及适用范围

核对官方源文件 MD5，剔除作者标记的 3,054 个几何不一致条目和 README 中的 11 个优化困难条目。
源频率必须全为正。所选结构全部通过原始/优化 SMILES 连接图一致性、XYZ 重新推断连接关系及手性、
分子式、有限坐标、最小原子间距、偶数电子检查，并成功构建 PySCF def2-SVP 分子对象。
原始 GDB SMILES 常省略手性；不会仅因优化后 SMILES 增加手性标记就丢弃该分子，
但最终所选 XYZ 推断的立体化学必须与优化结构 SMILES 一致。

本次没有运行 SCF、稳定性分析、CCSD(T) 或重新优化，也没有生成 MD/扰动构型。
AO 数用于预算规模；坐标本身不是 def2-SVP 优化结果。
后续需要核验参考态稳定性、CCSD 收敛和相关性诊断后才能生成合格的 CCSD(T) 标签。
如果后续每个分子采样 20 个构型，可形成 10,000 个标注候选；当前交付只有 500 个种子结构。

## 结构总览

''' + overview + '''

## 来源与复现

[QM9 原始数据](https://doi.org/10.6084/m9.figshare.978904_D12)，
Ramakrishnan et al., Scientific Data 1, 140022 (2014),
[DOI:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)。
官方数据条目许可为 CC0，元数据已保存。

Python 依赖：NumPy、RDKit、PySCF、Pillow；实际版本见 manifest。
将 manifest 列出的 3 个官方源文件和 article-1057646/1057644/1057641.json 放在源目录，
运行 `python select_seeds.py --source-dir /path/to/source`。
输入的 pilot40 清单默认从本包 `provenance/pilot40_selection.json` 读取。
可选 `--cache-dir /path/to/cache`；筛选规则或来源变更后应使用空缓存目录。
脚本只准备结构，不提交计算。
'''
    (out / 'README.md').write_text(readme)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[3] / 'datasets' / 'qm9_ccsdt_seeds_500')
    parser.add_argument('--pilot-selection', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    pilot_path = args.pilot_selection or out / 'provenance' / 'pilot40_selection.json'
    pilot = json.loads(pilot_path.read_text())
    pinned = {r['qm9_id']: r for r in pilot}
    assert len(pinned) == 40
    sources = []
    for name, (aid, fid, md5) in FILES.items():
        path = args.source_dir / name
        assert sha(path, 'md5') == md5, name
        sources.append(dict(filename=name, article_id=aid, file_id=fid,
                            url='https://ndownloader.figshare.com/files/' + fid,
                            md5=md5, sha256=sha(path)))
    excluded = {int(m[1]) for line in (args.source_dir/'uncharacterized.txt').read_text().splitlines()
                if (m := re.match(r'^\s*(\d+)\s', line))}
    assert len(excluded) == 3054
    candidates, screening = screen_sources(args.source_dir, excluded, args.cache_dir)
    print(json.dumps(screening, indent=2), flush=True)
    selected, failures = select(candidates, pinned)
    manifest = write_results(out, selected, pinned, sources, screening, failures)
    dump(out / 'provenance' / 'pilot40_selection.json', pilot)
    for name in ['readme.txt', 'uncharacterized.txt', 'article-1057646.json', 'article-1057644.json', 'article-1057641.json']:
        shutil.copy2(args.source_dir / name, out / 'provenance' / name)
    files = sorted(f for f in out.rglob('*') if f.is_file() and f.name != 'SHA256SUMS' and '__pycache__' not in f.parts)
    (out/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(out)}\n' for f in files))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f, f.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({k: manifest[k] for k in ['molecule_count', 'structure_count', 'heavy_atom_counts',
                                              'ao_def2_svp_range', 'element_molecule_counts',
                                              'functional_tag_counts', 'geometry_selection_rejections']}, indent=2), flush=True)
    print(f'Archive verified: {archive}, {archive.stat().st_size:,} bytes', flush=True)


if __name__ == '__main__':
    main()
