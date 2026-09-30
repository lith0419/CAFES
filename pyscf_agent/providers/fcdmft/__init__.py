"""Optional fcDMFT provider for periodic GW and DMFT workflows."""

from .adapter import run_gw_dmft, run_hf_dmft
from .availability import fcdmft_availability, fcdmft_is_available
from .contracts import (
    DEFAULT_HF_DMFT_OPTIONS,
    DEFAULT_PERIODIC_GW_OPTIONS,
    GW_DMFT_RESULT_SCHEMA,
    HF_DMFT_ARRAYS_SCHEMA,
    HF_DMFT_RESULT_SCHEMA,
    PERIODIC_GW_ARRAYS_SCHEMA,
    PERIODIC_GW_RESULT_SCHEMA,
    load_hf_dmft_inputs,
    normalize_gw_dmft_options,
    normalize_hf_dmft_options,
    normalize_periodic_gw_options,
    validate_gw_dmft_request,
    validate_hf_dmft_request,
    validate_periodic_gw_reference,
)
from .gw import run_local_gw_double_counting, run_periodic_gw
from .preparation import (
    prepare_periodic_fcdmft_subspace,
    prepare_periodic_hf_subspace,
    validate_fcdmft_preparation_request,
    validate_hf_dmft_preparation_request,
)

__all__ = [
    'DEFAULT_HF_DMFT_OPTIONS',
    'DEFAULT_PERIODIC_GW_OPTIONS',
    'GW_DMFT_RESULT_SCHEMA',
    'HF_DMFT_ARRAYS_SCHEMA',
    'HF_DMFT_RESULT_SCHEMA',
    'PERIODIC_GW_ARRAYS_SCHEMA',
    'PERIODIC_GW_RESULT_SCHEMA',
    'fcdmft_availability',
    'fcdmft_is_available',
    'load_hf_dmft_inputs',
    'normalize_gw_dmft_options',
    'normalize_hf_dmft_options',
    'normalize_periodic_gw_options',
    'prepare_periodic_fcdmft_subspace',
    'prepare_periodic_hf_subspace',
    'run_gw_dmft',
    'run_hf_dmft',
    'run_local_gw_double_counting',
    'run_periodic_gw',
    'validate_fcdmft_preparation_request',
    'validate_gw_dmft_request',
    'validate_hf_dmft_preparation_request',
    'validate_hf_dmft_request',
    'validate_periodic_gw_reference',
]
