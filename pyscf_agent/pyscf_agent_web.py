"""Compatibility import; implementation lives in pyscf_agent.web.server."""

import sys
from importlib import import_module

_implementation = import_module('pyscf_agent.web.server')
if __name__ == '__main__':
    raise SystemExit(_implementation.serve())
sys.modules[__name__] = _implementation
