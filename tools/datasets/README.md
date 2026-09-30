# Dataset preparation

The reusable QM9 preparation tools live in `qm9/`:

```sh
python -m tools.datasets.qm9.pilot --source-dir /path/to/qm9-source
python -m tools.datasets.qm9.diverse --source-dir /path/to/qm9-source
```

The pilot retains its 40 named chemical motifs. The diverse policy retains its
500-molecule quotas, screening rules, and random seed. They share numeric parsing,
checksums, canonicalization, and official source identities. Running these commands
requires the original optional RDKit/PySCF dependencies. Source datasets,
selected geometries and calculation outputs are maintained separately from
this public source repository.
