/-
  AxiomCheck.lean: one-command axiom audit of the formal development.

  `PCESurrogate.lean` proves eight lemmas covering the seven numbered results T1 to T7
  of the manuscript (T3 is split into T3a positivity and T3b normalisation), plus the
  auxiliary lemma `simplex_l1_le_two` used by the T5 proof. This file re-checks every
  one of them in a single run: `#print axioms` reports the axioms a declaration
  actually rests on, and it reports `sorryAx` for anything left unproved, so it is the
  direct check of both claims made in the manuscript (no `sorry`, standard axioms only).

  How to run (from the `lean/` directory, after the prerequisites in README.md):

      lake build                    -- builds PCESurrogate against Mathlib
      lake env lean AxiomCheck.lean -- prints one axiom line per declaration

  `lake build` alone does not compile this file: the default target is the
  `PCESurrogate` library root, so adding this audit cannot affect the library build.

  Expected output: one line per declaration, of the form

      'pce_error_tendsto_zero' depends on axioms: [propext, Classical.choice, Quot.sound]

  A declaration may list a subset of those three. Anything else, in particular any
  mention of `sorryAx` or of an axiom outside that list, would contradict the
  `sorry`-free claim.
-/

import PCESurrogate

-- T1: PCE truncation error converges to zero (conditional on the Sobolev rate)
#print axioms pce_error_tendsto_zero

-- T2: Sobol index convergence from l^2 coefficient convergence
#print axioms sobol_convergence_from_L2

-- T3a: softmax strict positivity
#print axioms softmax_positive

-- T3b: softmax normalisation identity
#print axioms softmax_sums_to_one

-- T4: Lipschitz uncertainty propagation
#print axioms uncertainty_propagation

-- T5: softmax l-infinity to l-1 Lipschitz bound, with constant 2
#print axioms softmax_lipschitz

-- Auxiliary lemma used by the T5 proof: the l-1 diameter of the simplex is 2
#print axioms simplex_l1_le_two

-- T6: softmax is invariant to a uniform shift of the logits
#print axioms softmax_shift_invariant

-- T7: decision value-variability is bounded by the fragility
#print axioms value_diff_le_l1
