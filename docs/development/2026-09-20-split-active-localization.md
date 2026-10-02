# Occupation-partitioned active-orbital localization

Active-space Boys/PM localization now accepts an explicit
`orbital_processing.localization_occupation_thresholds` list. The empty default
retains whole-block localization. `[1.0]` separates virtual-like and
occupied-like subspaces; `[0.1, 1.9]` additionally isolates intermediate
occupations. Thresholds are strictly increasing inside (0,2), with equality
assigned to the lower interval, following the partition convention in block2's
UNO `sort_orbitals` example.

The runtime projects the spin-summed SCF AO density into the actual chosen
active block, diagonalizes that small density, and localizes each occupation
interval independently. It does not infer occupations from MO positions or
from diagonals in an arbitrary localized basis. It does not require a 2-RDM,
select a new active space or change the electron count. These reference
occupations are distinct from correlated NOONs.

```json
{
  "orbital_processing": {
    "localization_method": "pipek_mezey",
    "localization_scope": "active_space",
    "localization_occupation_thresholds": [1.0],
    "orbital_ordering": "fiedler"
  }
}
```

The existing CAS initialization calls the shared numerical implementation in
`pyscf_agent/backend/correlation/orbital_localization.py`. Core and external
columns are preserved. The result records the density source, reference
occupations, group indices/counts, localization convergence observations, and
orthonormality/subspace checks. Indices refer to the descending-occupation
natural basis before within-group localization and any subsequent Fiedler
permutation. Empty and singleton groups require no optimization.

Boys/PM uses PySCF's existing optimizer and defaults. This follows Huanchen's
split-subspace suggestion; it is not a reproduction of block2's separate
`pmloc` optimizer. Fiedler remains the provider's existing ordering operation.
For CASSCF the partition is applied only to initial orbitals, and the chosen
ordering remains fixed during internal solver calls. No artificial constraint
is imposed on subsequent orbital optimization. Compatible joint orbital/MPS
restarts retain their checkpoint basis instead of relocalizing it.

The field is available through the TaskSpec and existing structured request
interfaces. A dedicated WebUI form control was not added. Registry metadata,
Wiki source/editorial guidance and the packaged runtime Wiki agree.

Validation: 171 provider, contract, Registry, workflow and numerical tests
passed. Added numerical checks cover density-subspace preservation after an
arbitrary initial rotation, FCI energy invariance, multiple/empty intervals,
reference-density validity and unchanged inactive/external orbitals. The
public UHF-reference DMRG-CASSCF continuation test now exercises split PM and
Fiedler together. After adding the final finite-output check, all six dedicated
localization tests passed again.

The authorized Amarel campaign is recorded in
the lutein comparison report (author archive: `reports/lutein-split-localization-2026-09-20.md`).
