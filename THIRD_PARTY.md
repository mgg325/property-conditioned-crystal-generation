# Third-party software and model assets

| Resource | Upstream / version evidence | Use in this work | Release treatment |
|---|---|---|---|
| MatterGen | [Microsoft MatterGen](https://github.com/microsoft/mattergen), git `a245cf2`; package metadata `1.0.3`; MIT | adapter backbone, CFG sampling, thermodynamic evaluator, reward-guided updates | not redistributed; only project-specific overlay files are included under `src/mattergen/patches/` and `src/mattergen/dielectric_rl/` |
| Crystalite | [joshrosie/crystalite](https://github.com/joshrosie/crystalite), git `e0cffad`; upstream `LICENSE` is MIT | dielectric-conditioned Transformer generation | not redistributed wholesale; final project-modified/required source subset retained under `src/crystalite/final/` |
| AnisoNet | [Virtual Atoms Lab AnisoNet](https://github.com/virtualatoms/AnisoNet), git `8d06f5f`; MIT | dielectric labels and prediction | model checkpoint and raw dataset excluded; project preparation/prediction helpers retained |
| MatterSim | Python package, MatterGen metadata requires `mattersim>=1.1` | structural relaxation / energy evaluation | external installation and model assets required |
| pymatgen / SMACT | external Python packages | structure parsing, matching, chemistry checks | installed dependencies, not redistributed |

Pretrained MatterGen, Crystalite, AnisoNet, and MatterSim weights are excluded. Obtain them under their upstream terms. This release contains no assertion that any dataset or checkpoint may be redistributed.

Modified MatterGen files are represented by the overlay under `src/mattergen/patches/mattergen/`. Modified Crystalite files are retained in the upstream-relative layout under `src/crystalite/final/src/`. The root MIT licence applies only to project-specific code; upstream licence notices remain applicable to their respective derived files.

Raw dielectric data, thermodynamic reference data, and model assets are not included. Their acquisition and redistribution terms must be confirmed from the relevant provider before reuse or redistribution.

Verified upstream licence texts are retained in `licenses/MATTERGEN_LICENSE`, `licenses/CRYSTALITE_LICENSE`, and `licenses/ANISONET_LICENSE`.
