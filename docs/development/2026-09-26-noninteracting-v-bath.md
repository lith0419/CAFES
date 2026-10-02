# Finite-graph noninteracting bath with intersite V

Explicit `interacting_bath=false` is now honored for U+V models. The default
remains true. V still selects the finite-graph execution path.

This implementation uses a density-updated environment mean field. At each
outer iteration, D is the auxiliary lattice density, F[D]=h+JK_full[D], P is
the fragment embedding basis, and g_imp retains only ERIs with all four
indices in the physical impurity. The solving Hamiltonian is

    h_emb = P† F[D] P - JK_imp[P† D P] + P† u_env P
    g_emb = g_imp

Here u is the fitted auxiliary potential minus the fixed seed HF baseline.
Only the current physical impurity block is removed from u_env. The baseline
is not added a second time. This is an explicitly specified NIB convention;
it should not be described as the unique definition of NIB for nonlocal V.

Native libDMET transform_imp refers to an entire cell. A finite graph uses one
cell for the whole system, so the native local-potential subtraction would
remove the environment correction too. For nonzero V, the adapter masks the
physical fragment block and bypasses that whole-cell subtraction. Existing
V=0 branches are unchanged.

Energy is evaluated in the NIB solver's correlated state using a separately
constructed full physical U+V embedding operator and libDMET's democratic
impurity-index partition. The physical frozen-core JK and its half weighting
are computed separately from the omitted-interaction mean field used in the
solving Hamiltonian. Neither the fitted u nor the fitted chemical-potential
shift enters this energy operator. This requires projecting the full ERIs for
energy even though the solver only retains impurity ERIs.

Validation: tests/pyscf_agent/test_noninteracting_v_bath.py checks explicit
false at V!=0, retained impurity V and zero bath ERIs, the Fock/JK/correction
identity with unequal alpha/beta densities, determinant physical energy summed
across two fragments, isolation of energy and solver mean fields, recovery of
the original Hamiltonian in the full-fragment limit, and one outer iteration
using restricted FCI, unrestricted FCI, and unrestricted CCSD. All four tests
pass against local libDMET 0.5. Forty-nine existing contract and selected
execution regressions also pass. These checks do not establish outer DMET
convergence or accuracy at strong coupling.

Comparison prepared separately: periodic 4x4 honeycomb, 2x2-cell impurities,
U=1, V=0.1, t=-1, half filling, unrestricted AF seed, beta=1000, CCSD. Compare
interacting and noninteracting baths with otherwise identical settings.
