import Lake
open Lake DSL

package pce_surrogate where
  moreLeanArgs := #["-DautoImplicit=false"]
  moreServerOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib PCESurrogate where
  roots := #[`PCESurrogate]

require mathlib from ".." / ".." / "mathlib4"
