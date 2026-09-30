"""Compatibility command; use pyscf_agent.cli for new integrations."""
import sys
from importlib import import_module

_implementation = import_module('pyscf_agent.cli')
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
