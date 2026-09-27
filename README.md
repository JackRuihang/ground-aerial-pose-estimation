# Ground-to-Aerial Pose Estimation: Differentiable LM Solver

The differentiable 3DoF Levenberg–Marquardt solver developed in Ruihang Jiang's
master's thesis on weakly supervised ground-to-aerial feature matching at EPFL.
It estimates planar rotation and translation from weighted geometric correspondences
and supports backpropagation through the optimization steps.

**VITA Lab, EPFL · Jul–Dec 2025**  
Supervisor: Prof. Alexandre Alahi · Advisor: Dr. Zimin Xia

## Scope of this release
This repository publishes the independently implemented **pose solver**, with a
synthetic example and gradient check. The full image-matching/training pipeline,
third-party feature extractors, datasets and learned checkpoints are not included.
The repository does not claim end-to-end reproduction of the thesis experiments.

## Geometric formulation
For ground-view directions `dP` and corresponding aerial-plane points `P`, the solver
estimates `[theta, tx, ty]`. Each residual measures the component of a translated
aerial point perpendicular to the rotated ground-view ray. The implementation uses
an analytic Jacobian, damped normal equations and iterative angular wrapping.
Optional weights multiply the residual and its Jacobian; their effective squared-error
weight is consequently the square of the supplied value.

## Installation and example
Requires Python and PyTorch; install the appropriate PyTorch build for your system.

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

The smoke test creates synthetic correspondences from a known camera pose, checks
recovery within a numerical tolerance, then checks finite gradients with respect to
the point inputs. It runs on CPU and needs no dataset, weights or feature backbone.

```python
from lib.models.LMModule import LMModule

solver = LMModule(n_iter=12, device="cpu")
# directions and points: floating tensors shaped [batch, matches, 2]
# Use consistent geometric coordinate units; theta is in radians.
pose = solver(directions, points, weights=None, init_params=None)
```

Construct the solver on the intended device; this historical implementation stores
one geometric tensor outside registered buffers. Global convergence is not guaranteed:
initialization, correspondence geometry, ambiguity and outliers affect the estimate.

## Thesis context and reported results
The broader thesis investigates pose-supervised local matching, dense correspondence
scores and differentiable optimization between ground and aerial imagery.
Its four-pair sanity check and quarter-scale VIGOR experiment are separate from this
synthetic solver test. Table 5.1 reports **10.32 m** mean localization error and **24°**
mean orientation error for the project experiment (same area, unknown orientation).
The thesis discusses overfitting; these subset results are not directly comparable
with full-dataset baselines and are not a state-of-the-art claim.

## Files
- `lib/models/LMModule.py`: original differentiable solver.
- `smoke_test.py`: synthetic pose-recovery and backward-gradient test.
- `VALIDATION.md`: tested environment and limits.
- `THIRD_PARTY.md`: excluded components and attribution for the broader thesis.

## Acknowledgements and license
Thanks to Alexandre Alahi and Zimin Xia for supervision. The broader project uses
MicKey/DINOv2 features and DUSt3R/MASt3R-related infrastructure; those components are
not redistributed here. The solver and accompanying release utilities are MIT licensed.
No rights to external models, datasets or third-party implementations are granted.
