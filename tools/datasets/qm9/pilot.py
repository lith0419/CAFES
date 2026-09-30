#!/usr/bin/env python3
"""Select a curated pilot from the original QM9 archive; run no calculations.

Requires numpy, RDKit and PySCF. Source directory must contain the three
official Figshare files and article metadata named in SOURCE_FILES below.
Usage: python select_seeds.py --source-dir ../../tmp/qm9-source
"""
from __future__ import annotations

from tools.datasets.qm9.common import digest, numeric as number, canonical, OFFICIAL_SOURCE_FILES as SOURCE_FILES, DIFFICULT_QM9_IDS as DIFFICULT_IDS

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, rdDetermineBonds, rdMolDescriptors
from pyscf import gto
import pyscf


# Named chemical motifs for an interpretable pilot, not a random QM9 sample.
# English names are display labels; SMILES and original QM9 IDs define identity.
TARGETS = [
    ('methane', '甲烷', 'C', 'small', '烷烃；最小运行检查'),
    ('water', '水', 'O', 'small', '含氧小分子；最小运行检查'),
    ('methanol', '甲醇', 'CO', 'small', '醇'),
    ('ammonia', '氨', 'N', 'small', '含氮小分子；最小运行检查'),
    ('acetylene', '乙炔', 'C#C', 'small', '碳碳三键'),
    ('ethanol', '乙醇', 'CCO', 'small', '醇；构象自由度'),
    ('acetonitrile', '乙腈', 'CC#N', 'small', '腈；三键'),
    ('acetaldehyde', '乙醛', 'CC=O', 'small', '醛；羰基'),
    ('acetone', '丙酮', 'CC(=O)C', 'medium', '酮'),
    ('urea', '尿素', 'NC(N)=O', 'medium', '双氮羰基；酰胺对照'),
    ('acetamide', '乙酰胺', 'CC(N)=O', 'medium', '酰胺'),
    ('isobutane', '异丁烷', 'CC(C)C', 'medium', '支链烷烃'),
    ('1-propanol', '1-丙醇', 'CCCO', 'medium', '醇；柔性链'),
    ('methyl acetate', '乙酸甲酯', 'COC(C)=O', 'medium', '酯'),
    ('diethyl ether', '乙醚', 'CCOCC', 'medium', '醚；柔性链'),
    ('butanenitrile', '丁腈', 'CCCC#N', 'medium', '腈；链长对照'),
    ('tetrahydrofuran', '四氢呋喃', 'C1CCOC1', 'medium', '饱和含氧五元环'),
    ('pyrrole', '吡咯', 'c1cc[nH]c1', 'medium', '芳香含氮五元环'),
    ('furan', '呋喃', 'c1ccoc1', 'medium', '芳香含氧五元环'),
    ('tetrafluoromethane', '四氟甲烷', 'FC(F)(F)F', 'medium', '多氟取代；无氢分子'),
    ('n-hexane', '正己烷', 'CCCCCC', 'large', '长链烷烃；柔性'),
    ('cyclohexane', '环己烷', 'C1CCCCC1', 'large', '饱和六元碳环'),
    ('benzene', '苯', 'c1ccccc1', 'large', '芳香碳环'),
    ('pyridine', '吡啶', 'c1ccncc1', 'large', '芳香六元氮杂环'),
    ('pyrimidine', '嘧啶', 'c1cncnc1', 'large', '双氮芳香杂环'),
    ('2-pyrrolidone', '2-吡咯烷酮', 'O=C1CCCN1', 'large', '五元内酰胺'),
    ('1,4-dioxane', '1,4-二氧六环', 'C1COCCO1', 'large', '饱和双氧六元环'),
    ('phenol', '苯酚', 'Oc1ccccc1', 'large', '芳香羟基'),
    ('aniline', '苯胺', 'Nc1ccccc1', 'large', '芳香胺'),
    ('fluorobenzene', '氟苯', 'Fc1ccccc1', 'large', '芳香 C–F'),
    ('cyclohexanone', '环己酮', 'O=C1CCCCC1', 'large', '环酮'),
    ('ethyl acetate', '乙酸乙酯', 'CCOC(C)=O', 'large', '酯；链长对照'),
    ('ethylbenzene', '乙苯', 'CCc1ccccc1', 'stress', '芳香环与柔性侧链'),
    ('anisole', '苯甲醚', 'COc1ccccc1', 'stress', '芳香醚'),
    ('acetophenone', '苯乙酮', 'CC(=O)c1ccccc1', 'stress', '芳香酮；9 重原子'),
    ('benzamide', '苯甲酰胺', 'NC(=O)c1ccccc1', 'stress', '芳香酰胺；9 重原子'),
    ('ethyl butyrate', '丁酸乙酯', 'CCCC(=O)OCC', 'stress', '柔性脂肪族酯'),
    ('1,4-difluorobenzene', '1,4-二氟苯', 'Fc1ccc(F)cc1', 'stress', '多氟芳香取代'),
    ('2-methoxyethyl acetate', '乙酸-2-甲氧基乙酯', 'COCCOC(C)=O', 'stress', '醚与酯；多可旋转键'),
    ('N-ethyl-2-pyrrolidone', 'N-乙基-2-吡咯烷酮', 'CCN1CCCC1=O', 'stress', '内酰胺与柔性侧链'),
]

GROUP_LABELS = {'small': '小分子检查', 'medium': '4–5 重原子', 'large': '6–7 重原子', 'stress': '8–9 重原子'}








def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[3] / 'datasets' / 'qm9_ccsdt_pilot_40')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source_info = []
    for filename, (aid, fid, md5) in SOURCE_FILES.items():
        path = args.source_dir / filename
        if digest(path, 'md5') != md5:
            raise ValueError('Official MD5 mismatch: ' + filename)
        source_info.append({'filename': filename, 'article_id': aid, 'file_id': fid,
                            'url': 'https://ndownloader.figshare.com/files/' + fid,
                            'md5': md5, 'sha256': digest(path)})

    excluded = {int(m.group(1)) for line in (args.source_dir / 'uncharacterized.txt').read_text().splitlines()
                if (m := re.match(r'^\s*(\d+)\s', line))}
    assert len(excluded) == 3054, len(excluded)
    targets = {canonical(Chem.MolFromSmiles(row[2])): row for row in TARGETS}
    assert len(targets) == len(TARGETS) == 40
    element_counts = {tuple(sorted(Counter(a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(row[2])).GetAtoms()).items()))
                      for row in TARGETS}
    found = {}
    scanned = 0
    with tarfile.open(args.source_dir / 'dsgdb9nsd.xyz.tar.bz2', 'r|bz2') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.xyz'):
                continue
            raw = archive.extractfile(member).read()
            lines = raw.decode().splitlines()
            natoms = int(lines[0])
            qid = int(lines[1].split()[1])
            scanned += 1
            if qid in excluded or qid in DIFFICULT_IDS:
                continue
            counts = tuple(sorted(Counter(line.split()[0] for line in lines[2:2+natoms]).items()))
            if counts not in element_counts:
                continue
            smiles = lines[natoms+3].split()
            mol = Chem.MolFromSmiles(smiles[0])
            if mol is None:
                continue
            key = canonical(mol)
            if key not in targets:
                continue
            optimized = Chem.MolFromSmiles(smiles[1])
            if optimized is None or canonical(optimized) != key:
                raise ValueError(f'Geometry SMILES mismatch for selected molecule {qid}')
            if key in found:
                # Lowest original ID is the deterministic tie-break, not energy.
                if qid > found[key]['qm9_id']:
                    continue
            found[key] = {'qm9_id': qid, 'member': member.name, 'raw': raw, 'lines': lines,
                          'mol': mol, 'smiles_gdb9': smiles[0], 'smiles_optimized': smiles[1]}

    missing = [row[0] for key, row in targets.items() if key not in found]
    print(json.dumps({'scanned': scanned, 'found': len(found), 'missing': missing}, ensure_ascii=False), flush=True)
    if missing:
        raise RuntimeError('Selected molecules absent from eligible QM9 records: ' + ', '.join(missing))
    assert scanned == 133885

    for directory in ['xyz', 'source_originals', 'provenance']:
        (out / directory).mkdir(exist_ok=True)
    for filename in ['uncharacterized.txt', 'readme.txt']:
        shutil.copy2(args.source_dir / filename, out / 'provenance' / filename)
    for aid in ['1057646', '1057644', '1057641']:
        shutil.copy2(args.source_dir / f'article-{aid}.json', out / 'provenance' / f'article-{aid}.json')

    records, seeds, all_xyz, draw_mols, legends = [], [], [], [], []
    for order, target in enumerate(TARGETS, 1):
        name, chinese, input_smiles, group, reason = target
        key = canonical(Chem.MolFromSmiles(input_smiles))
        item = found[key]
        qid, lines, mol = item['qm9_id'], item['lines'], item['mol']
        natoms = int(lines[0])
        symbols = [line.split()[0] for line in lines[2:2+natoms]]
        positions = [[number(x) for x in line.split()[1:4]] for line in lines[2:2+natoms]]
        xyz_body = '\n'.join(' '.join(line.split()[:4]).replace('*^', 'e') for line in lines[2:2+natoms])
        qname = f'qm9_{qid:06d}'
        xyz = f'{natoms}\n{qname} | {name} | charge=0 spin=0 | Angstrom | original QM9 B3LYP/6-31G(2df,p) geometry\n{xyz_body}\n'
        atomic_numbers = [Chem.GetPeriodicTable().GetAtomicNumber(s) for s in symbols]
        nelectrons = sum(atomic_numbers)
        assert nelectrons % 2 == 0
        assert Chem.GetFormalCharge(mol) == 0
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
        assert len(Chem.GetMolFrags(mol)) == 1
        assert Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()) == Counter(symbols)
        coords = np.asarray(positions)
        assert np.isfinite(coords).all()
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        minimum_distance = float(distances[np.triu_indices(natoms, 1)].min())
        assert minimum_distance > 0.5
        xyz_mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(xyz_mol, charge=0, allowChargedFragments=True)
        assert canonical(xyz_mol) == key, (qid, canonical(xyz_mol), key)
        frequencies = [number(value) for value in lines[natoms+2].split()]
        assert frequencies and min(frequencies) > 0, (qid, min(frequencies))
        # Basis construction only: no SCF, integrals, optimization or CC execution.
        pmol = gto.M(atom=list(zip(symbols, positions)), unit='Angstrom', charge=0,
                     spin=0, basis='def2-svp', verbose=0)
        nheavy = sum(z > 1 for z in atomic_numbers)
        nao = pmol.nao_nr()
        assert pmol.nelectron == nelectrons
        nocc = nelectrons // 2
        nvir = nao - nocc
        source_hash = hashlib.sha256(item['raw']).hexdigest()
        xyz_path = out / 'xyz' / (qname + '.xyz')
        xyz_path.write_text(xyz)
        (out / 'source_originals' / Path(item['member']).name).write_bytes(item['raw'])
        row = dict(selection_order=order, molecule_id=qname, qm9_id=qid,
                   name=name, name_zh=chinese, group=group, selection_reason=reason,
                   formula=rdMolDescriptors.CalcMolFormula(mol), smiles=key,
                   smiles_gdb9=item['smiles_gdb9'], smiles_optimized=item['smiles_optimized'],
                   n_atoms=natoms, n_heavy_atoms=nheavy, n_electrons=nelectrons,
                   n_ao_def2_svp=nao, n_occ_rhf=nocc, n_vir_rhf=nvir,
                   suggested_frozen_core_orbitals=nheavy,
                   n_occ_if_frozen_1s=nocc-nheavy,
                   ring_count=rdMolDescriptors.CalcNumRings(mol),
                   rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
                   min_distance_angstrom=minimum_distance,
                   min_source_frequency_cm1=min(frequencies), charge=0, spin=0,
                   coordinate_unit='Angstrom', xyz_path='xyz/' + xyz_path.name,
                   source_member=item['member'], source_sha256=source_hash,
                   excluded_by_qm9=False, geometry_connectivity_verified=True,
                   ccsdt_suitability='not_assessed')
        records.append(row)
        seeds.append(dict(molecule_id=qname, geometry_id='qm9_optimized_seed',
                          atomic_numbers=atomic_numbers, positions=positions,
                          coordinate_unit='Angstrom', charge=0, spin=0,
                          source={'dataset': 'QM9', 'qm9_id': qid,
                                  'geometry_level': 'B3LYP/6-31G(2df,p)',
                                  'archive_member': item['member'], 'sha256': source_hash,
                                  'doi': '10.6084/m9.figshare.978904_D12'}))
        all_xyz.append(xyz)
        draw_mols.append(mol)
        legends.append(f'{order:02d}  {name}\nQM9 {qid} | {nheavy} heavy | {nao} AO')

    (out / 'seeds.xyz').write_text(''.join(all_xyz))
    (out / 'seed_geometries.json').write_text(json.dumps({'seed_geometries': seeds}, indent=2) + '\n')
    (out / 'selection.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    with (out / 'selection.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    svg = Draw.MolsToGridImage(draw_mols, molsPerRow=5, subImgSize=(360, 260), legends=legends,
                              useSVG=True, legendFontSize=20, legendFraction=0.22)
    (out / 'molecules.svg').write_text(svg)
    img = Draw.MolsToGridImage(draw_mols, molsPerRow=5, subImgSize=(360, 260), legends=legends,
                              legendFontSize=20, legendFraction=0.22)
    img.save(out / 'molecules.png')
    manifest = {'schema': 'qm9-curated-pilot-v1', 'selection_count': len(records),
                'selection_method': 'Curated named chemical motifs; exact canonical SMILES matching; lowest QM9 ID tie-break.',
                'selection_is_statistically_representative': False,
                'source_archive_records_scanned': scanned, 'official_exclusion_count': len(excluded),
                'additional_difficult_geometry_ids_excluded': sorted(DIFFICULT_IDS),
                'source_files': source_info,
                'source_geometry_level': 'B3LYP/6-31G(2df,p)',
                'coordinates_modified': False, 'coordinate_unit': 'Angstrom',
                'source_license': 'CC0 (Figshare article metadata)',
                'rdkit_version': rdkit.__version__, 'pyscf_version': pyscf.__version__,
                'heavy_atom_counts': dict(sorted(Counter(r['n_heavy_atoms'] for r in records).items())),
                'groups': dict(Counter(r['group'] for r in records)),
                'ao_def2_svp_range': [min(r['n_ao_def2_svp'] for r in records), max(r['n_ao_def2_svp'] for r in records)],
                'validation': {'source_md5': 'passed', 'original_vs_optimized_smiles': 'passed',
                               'xyz_connectivity_vs_smiles': 'passed', 'unique_smiles_and_ids': 'passed',
                               'finite_coordinates': 'passed', 'positive_source_frequencies': 'passed',
                               'closed_shell_electron_count_and_no_graph_radicals': 'passed',
                               'pyscf_def2_svp_molecule_build': 'passed'},
                'new_quantum_calculations_executed': 0,
                'ccsdt_suitability': 'Not established; requires reference stability and correlation diagnostics.',
                'sampled_non_equilibrium_structures': 0}
    assert len({r['qm9_id'] for r in records}) == len(records)
    assert len({r['smiles'] for r in records}) == len(records)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    table = ['|序号|QM9 ID|分子|分子式|重原子|def2-SVP AO|选择理由|',
             '|---:|---:|---|---|---:|---:|---|']
    for r in records:
        table.append(f"|{r['selection_order']}|{r['qm9_id']}|[{r['name_zh']}]({r['xyz_path']})|{r['formula']}|{r['n_heavy_atoms']}|{r['n_ao_def2_svp']}|{r['selection_reason']}|")
    readme = '''# QM9 初始结构：40 分子 CCSD(T) 试运行候选集

本目录包含 40 个不同分子的 **原始 QM9 优化几何**，每个分子 1 个结构。
用途是为后续 CCSD(T) 成本试跑和构型采样提供可追溯的起点。
这是按常见化学结构人工设计的试验集，不是 QM9 的随机或统计代表性子集。
分组为 8 个小分子、12 个 4–5 重原子分子、12 个 6–7 重原子分子、8 个 8–9 重原子分子。

![分子结构总览](molecules.png)

## 文件

- `xyz/`：40 个标准 XYZ 文件；保留原始原子顺序、方向和坐标，单位 Å。
- `seeds.xyz`：按选择序号排列的多帧 XYZ，40 帧分别属于 40 个分子。
- `seed_geometries.json`：现有 agent `MolecularGeometry` 格式；其中 `seed_geometries` 数组可用作种子输入。
- `selection.csv` / `selection.json`：名称、SMILES、QM9 ID、电子数、AO 数、官能团选择理由等。
- `molecules.svg` / `molecules.png`：2D 结构清单，图中的 AO 数是球谐 def2-SVP 基函数数。
- `source_originals/`：原始 QM9 扩展 XYZ 原文，保留频率、属性、SMILES 和 InChI。
- `provenance/`：官方排除清单、原始说明和 Figshare 元数据；`manifest.json` 保存源文件校验值与验证结果。

## 已做的验证

从官方归档扫描全部 133,885 条记录；核对 3 个源文件官方 MD5，排除官方列出的
3,054 个几何一致性失败条目，以及 README 列出的 11 个优化困难条目。
所选 40 个结构均通过原始/优化 SMILES 一致性、XYZ 重建连接关系、元素和氢原子数、
中性无自由基分子图、偶数电子、有限坐标、最短原子间距和源频率全为正的检查。
还用 PySCF 构建了 def2-SVP 分子对象以核对电子数和基函数数；没有运行 SCF 或 CC。

这些检查不等于验证 RHF 稳定性、弱相关性或 CCSD(T) 精度；后续仍需参考态与相关性诊断。
`suggested_frozen_core_orbitals` 仅表示若冻结每个 C/N/O/F 的 1s 轨道时的轨道数，未实际应用计算设置。

## 几何与计算标签的区别

种子坐标来自 **B3LYP/6-31G(2df,p)** 优化，未重新优化为 def2-SVP 或 CCSD(T) 几何。
清单中的 `n_ao_def2_svp` 仅用于未来试跑成本分层。
本次没有生成扰动/MD 构型，也没有生成任何新的能量或力标签。
这 40 个结构本身不是新的 QH9 数值复现或 CCSD(T) 数据集。

建议先覆盖全部 4 个尺寸组做单点试跑，再将每个分子扩成 3 个低冗余构型，共 120 个点。
比较平均/P95 耗时、内存、收敛及相关性诊断后，再预算更大的采样与标注任务。
若后续做学习实验，应按分子分组划分数据，避免同一分子的构型泄漏到不同集合。

## 结构清单

''' + '\n'.join(table) + '''

## 来源与复现

原始结构：[Figshare: Data for 133885 GDB-9 molecules](https://doi.org/10.6084/m9.figshare.978904_D12)。
数据论文：Ramakrishnan et al., *Scientific Data* **1**, 140022 (2014),
[DOI: 10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)。
官方文件条目的当前许可为 CC0，原始元数据随本目录保存。

依赖：Python、NumPy、RDKit 2025.09.6、PySCF（实际版本见 manifest）。
下载 `manifest.json` 中的三个源文件及对应 Figshare API 文章 JSON，放在同一 source 目录。
运行 `python select_seeds.py --source-dir /path/to/source --output-dir /path/to/output`。
脚本仅选择和验证结构，不提交任何量子化学任务。
'''
    (out / 'README.md').write_text(readme)
    print(json.dumps({k: manifest[k] for k in ['selection_count', 'groups', 'heavy_atom_counts', 'ao_def2_svp_range', 'validation']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
