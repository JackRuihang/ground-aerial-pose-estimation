# Validation

2026-09-27: Python 3.10, PyTorch 2.11.0+cu128; CPU execution, two threads.

- Syntax check: passed.
- Synthetic known-pose recovery: passed (absolute tolerance 1e-3).
- Finite backward gradients with respect to input points: passed.
- Real-image matching, full training and thesis results: not reproduced by this release.

Run `python smoke_test.py` from the repository root. No automatic weight downloads occur.
