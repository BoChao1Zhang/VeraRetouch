# Retained source support

This package is not a databuild producer. The only production databuild entry point is:

```bash
python -m construct.agent run --config /absolute/path/to/databuild.toml
```

The retained modules provide the inputs and shared services used by that command:

- `ingest.py`, `ingest_hires.py`, and `db.py`: PostgreSQL-backed image inventory support.
- `caption_subjects.py`: strict OpenAI Responses caption and subject metadata.
- `sam3_subject_instances.py`: the instance-level `subject.json + subject.png` protocol.
- `iaa.py`: OneAlign candidate scoring.
- `lr_render.py`: Lightroom farm support for non-canonical consumers only. Canonical
  databuild cannot call the farm backend.

Source-QA admission gates, concept masks, calibration, pilot workflows, manual
backfills, and the review UI were removed. Source-QA verdicts do not filter the
canonical source pool.
