#!/usr/bin/env python3
"""Source-checkout shim for the installed ``pyscf-agent-configure`` CLI."""

import sys
from pyscf_agent import configure as _implementation


if __name__ == '__main__':
    raise SystemExit(_implementation.main())

# Preserve source-checkout imports and patches while owning no configuration.
sys.modules[__name__] = _implementation
