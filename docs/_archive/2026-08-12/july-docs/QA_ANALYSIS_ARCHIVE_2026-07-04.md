# QA Analysis Archive

Date: 2026-07-04

The QA analysis branch is archived and should no longer be treated as the main
implementation direction for IAA scoring.

Archived local branch name:

- `archive/qa-analysis-datagen-v2`

Scope retained for reference:

- source QA database and rendering experiments
- vLLM QA-judge prompts and gating logic
- preset analysis/calibration utilities

Current direction:

- Use dedicated IAA models for aesthetic scoring.
- Primary IAA model pair: Charm + ArtiMuse.
- Keep QA analysis outputs as historical evidence only; do not use them as the
  default scoring branch.
