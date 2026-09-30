# CAFES Distribution

CAFES has three distinct distribution products:

- `python -m build` creates the formal Python wheel and sdist release.
- `package.sh` creates a timestamped source snapshot of the current working
  tree for research and development transfer; it is not a Python sdist.
  Local QM9 numerical upload packages and manuscript material under
  `docs/si/` and `manuscript/` are excluded.
- `offline/package.sh` creates a self-contained Linux x86_64 HPC bundle from a
  fully pinned, version-specific target-platform lock.
- `offline/install.sh` is the installer template copied into that bundle as
  `install_offline.sh`; it verifies all wheel hashes before installation.
- `offline/requirements-linux-x86_64.in` records intentional direct inputs;
  `offline/generate-locks.sh` resolves the complete Python-version locks.

Verified wheel-only locks currently cover CPython 3.10-3.14. Python 3.10-3.13
target manylinux2014/glibc 2.17+, while Python 3.14 targets glibc 2.28+ because
current h5py wheels no longer support the older baseline. Python 3.15 remains a
readiness target until its required scientific wheels are published.

See [Verification and distribution](../docs/guides/verification-and-distribution.md)
and [Installation](../docs/guides/installation.md) for commands and supported
platform boundaries.
