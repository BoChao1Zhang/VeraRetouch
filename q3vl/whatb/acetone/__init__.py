"""EPR-034 -- the public AceTone (arXiv:2604.00530) on the whatb V_what board.

Two rows, two environments, one measurement口径:

* **row C** (``whatb_ACETONE_tokenizer``) -- the LUT tokenizer's reconstruction
  of *our* GT LUTs: bank LUT -> 32³ (AceTone's own resampler) -> VQ-VAE encode
  -> 64 tokens -> decode -> our applier -> headline;
* **row A** (``whatb_ACETONE_pst``) -- ``AceTone-3B-PST-Preview``, with the GT
  after-image as the style reference, prompt and generation kwargs copied out of
  ``eval/predict_lut_ddp.py``.

External inference runs in ``/home/bc/data/external/acetone/venv`` and writes
LUTs to disk.  Every published number is computed back in the campaign
environment by :mod:`q3vl.whatb.acetone.scoring`, which calls
``q3vl.whatb.criteria`` and nothing of AceTone's.

Modules
-------
``bridge``   the only code that touches the external clone (import-safe in both
             environments)
``rowset``   ``A_rows`` -- the row set is EPR-033's row set
``scoring``  ``A_baseline`` / ``A_finite``, the per-sample rows and the board
``scripts``  the five entry points (export / VQ / PST / two publishers)
"""

__all__ = ["bridge", "rowset", "scoring"]
