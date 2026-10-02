# Dataset preparation

The reusable QM9 preparation tools live in `qm9/`:

```sh
python -m tools.datasets.qm9.pilot --source-dir /path/to/qm9-source
python -m tools.datasets.qm9.diverse --source-dir /path/to/qm9-source
```

The pilot retains its 40 named chemical motifs. The diverse policy retains its
500-molecule quotas, screening rules, and random seed. They share numeric parsing,
checksums, canonicalization, and official source identities. Existing
`select_seeds.py` files in the archived seed directories are frozen historical
reproduction snapshots, kept with their original delivery checksums. Use those snapshots to reproduce
the published selections; use the maintained tools here for subsequent work.
Running these commands requires the original optional RDKit/PySCF dependencies;
this refactor does not download the source archive or regenerate selections.

The selected geometries, provenance manifests, IDs, checksums and bulk datasets
were archived on Amarel on 2026-10-01 and removed from this checkout after
checksum verification. See [storage and restore locations](../../docs/guides/repository-storage.md).
Source archives, caches, and new calculation outputs belong outside the checkout.
Report-specific reproduction recipes stay
beside their reports; they are not part of the installed runtime API.
