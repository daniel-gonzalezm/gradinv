"""Coordinate-free algebra of sixth-order strain-gradient elasticity tensors.

Modules
-------
skeleton     GL(3) skeleton of the strain-gradient space V (form II).
algebra      BlockAlgebra: matrix units, block moduli, inverse, positive definiteness.
isotropic    isotropic algebra (dimension 6); constants of Lazar et al. (2022).
cubic        cubic algebra (dimension 15); constants of Lazar et al. (2022).
hexagonal    hexagonal algebra (dimension 32); extra modulus; Auffray et al. (2013).
index_forms  level (b) expansions of the units in delta and structure tensors.
averages     Voigt, Reuss and Hill orientation averages in block moduli.
concentrator Solyaev's (2023) concentration tensor in block form; six invariants.
eshelby      averaged Eshelby tensors of the sphere (first gradient, isotropic medium).
selfconsistent  Solyaev's (2023) self-consistent scheme (cubic grains, Helmholtz medium).
lengths      block lengths relative to the Helmholtz form; Lazar-Po length.
forms        conversion between Mindlin's form I and form II variables.
"""

__version__ = "1.0.0"
