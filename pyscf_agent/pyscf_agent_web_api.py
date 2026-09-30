"""Compatibility import; implementation lives in pyscf_agent.web.api."""

import sys
from importlib import import_module

_implementation = import_module('pyscf_agent.web.api')
sys.modules[__name__] = _implementation
