"""GLUT from-scratch reproduction (A0) + per-LUT direct-fit engine (E1).

Spec: docs/IMPL_DOSSIER_2026-08-02.md section 2.2 (arXiv:2605.19889v1).
A0 copies the GLUT ORIGINAL initialization (mu grid, iso sigma=0.15 via
log-Cholesky, opacity raw=1.0 + clamp, M_i=I b_i=0, G=I g=0) to align the
45.47 dB / dE00 0.41 anchor.  Our own G=0 payload-gating changes belong to
the later RD arms, NOT here.
"""
