"""Compatibility import; implementation lives in pyscf_agent.web.ui."""

import sys
from importlib import import_module

_implementation = import_module('pyscf_agent.web.ui')
sys.modules[__name__] = _implementation
