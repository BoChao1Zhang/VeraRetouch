"""Arm registry for the EPR-018..023 batch: six names, one lazy-import contract.

The four live arms (``P1`` / ``P3prime`` / ``SHAPE3`` / ``UNIQ``) are branches
inside :class:`~q3vl.whereb.amort.model.AmortModel`.  The six new arms are not:
each one is a **self-contained module** that the model, the trainer and the
evaluator reach through the hooks declared here, so six agents can implement six
heads in parallel without ever touching a shared file.

    arm name      module                                   pre-registered column
    ------------  ---------------------------------------  ---------------------
    SEGSAM        q3vl.whereb.amort.segsam    (EPR-018)     segsam_fine
    SAMDEC        q3vl.whereb.amort.samdec    (EPR-019)     samdec_cand
    PRND          q3vl.whereb.amort.prnd      (EPR-020)     prnd_point_readout
    LIIF          q3vl.whereb.amort.liifhead  (EPR-021)     liif_grid_decode
    MATTE         q3vl.whereb.amort.matte     (EPR-022)     pix_readout
    CONDINST      q3vl.whereb.amort.condinst  (EPR-023)     condinst_pix_readout

The hooks
---------
Required (an arm module without all three fails to load, loudly)::

    ARM: str                       # must equal the registry name
    CRITERIA: tuple[str, ...]      # must equal ARM_CRITERIA[name]

    def build_head(*, in_dim: int, text_dim: int, args=None, **kw) -> nn.Module
        # constructed by AmortModel.__init__; `args` is the parsed argparse
        # namespace when the run script has one, else None.

    def forward(model, head, ctx: ArmContext) -> dict[str, Any]
        # called by AmortModel.forward_geo.  MUST return {"m_low": (gh, gw)}
        # -- the criterion path reads that key and nothing else -- plus whatever
        # the arm's own loss and diagnostic columns need.

    def compute_loss(model, out: dict, x, weights) -> AmortLoss
        # called by trainer.compute_micro_batch instead of the seven-term stack.

Required of the six new arms in particular::

    def loss_preregistration(args=None) -> dict
        # what `config/loss_preregistration.json` says for THIS arm.  The live
        # arms' record is the seven-term ST_LANG stack with
        # `dice_as_target: false`, which is FALSE for SEGSAM (dice 0.5),
        # SAMDEC (dice 1.0) and CONDINST (dice IS the target); a new arm that
        # let that file stand would publish a false pre-registration.
        # `run_amort_arm` refuses to start a new arm whose module has no such
        # hook rather than fall back to the live arms' form.

Optional (absent = "this arm does not use it")::

    def add_arguments(ap) -> None                  # register --<arm>-* flags
    def head_kwargs_from_args(args) -> dict        # -> build_head(**kw)
    def optimizer_spec(args) -> OptimizerSpec|None # per-arm optimizer
    def scheduler_kwargs(args, total_steps) -> dict
    def builder_kwargs(args) -> dict               # -> AmortBatchBuilder(**kw)
    def readout_spec(args) -> ReadoutSpec          # default: from --readout*
    def per_sample_row(model, out, x) -> dict      # extra per-sample eval cols
    def criteria_columns(rows) -> dict             # -> board["criteria_columns"]
    def train_stats(out, x) -> dict                # extra per-step stat columns

Why a registry and not six ``elif``s: ``model.py`` / ``trainer.py`` /
``evaluate.py`` are files all six agents would otherwise edit at the same time,
and a merge conflict in the dispatch chain is a silent way to run arm A's loss
under arm B's name.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "NEW_ARMS", "ARM_MODULES", "ARM_CRITERIA", "REQUIRED_HOOKS", "OPTIONAL_HOOKS",
    "RUN_REQUIRED_HOOKS", "ARM_KWARGS", "ARM_ARGS", "ARM_SETUP",
    "ArmContext", "load_arm", "arm_hook", "is_new_arm", "arm_criteria",
]

#: the six names ``--arm`` gains.  ORDER IS THE EPR ORDER, do not sort.
NEW_ARMS: tuple[str, ...] = ("SEGSAM", "SAMDEC", "PRND", "LIIF", "MATTE", "CONDINST")

ARM_MODULES: dict[str, str] = {
    "SEGSAM": "q3vl.whereb.amort.segsam",
    "SAMDEC": "q3vl.whereb.amort.samdec",
    "PRND": "q3vl.whereb.amort.prnd",
    "LIIF": "q3vl.whereb.amort.liifhead",
    "MATTE": "q3vl.whereb.amort.matte",
    "CONDINST": "q3vl.whereb.amort.condinst",
}

#: the pre-registered criterion column of each arm, quoted from its proposal.
#: :func:`q3vl.whereb.amort.evaluate.assert_criteria_ran` reads this table; an
#: arm whose board carries ``n = 0`` for its column cannot publish.
ARM_CRITERIA: dict[str, tuple[str, ...]] = {
    "SEGSAM": ("segsam_fine",),
    "SAMDEC": ("samdec_cand",),
    "PRND": ("prnd_point_readout",),
    "LIIF": ("liif_grid_decode",),
    "MATTE": ("pix_readout",),
    "CONDINST": ("condinst_pix_readout",),
}

#: Wrapper seam, mirroring ``uniq4.VARIANT4``: ``run_<arm>_arm.py`` parses its
#: own ``--<arm>-*`` flags, fills these, then delegates to
#: ``run_amort_arm.main(rest)``.  ``ARM_KWARGS`` reaches ``build_head`` and
#: ``ARM_ARGS`` reaches every ``*_from_args`` hook; ``ARM_SETUP`` is merged into
#: ``run_setup.json``.  Module-level state is not elegant, but it is the seam
#: this campaign's entries already use and the sha256 freeze covers it.
ARM_KWARGS: dict[str, Any] = {}
ARM_ARGS: Any = None
ARM_SETUP: dict[str, Any] = {}

REQUIRED_HOOKS: tuple[str, ...] = ("build_head", "forward", "compute_loss")
OPTIONAL_HOOKS: tuple[str, ...] = (
    "add_arguments", "head_kwargs_from_args", "optimizer_spec",
    "scheduler_kwargs", "builder_kwargs", "readout_spec", "per_sample_row",
    "criteria_columns", "train_stats", "loss_preregistration",
)

#: hooks that are optional to the *contract* (an arm module is importable
#: without them) but that ``run_amort_arm`` refuses to run a NEW arm without.
#: Kept out of ``REQUIRED_HOOKS`` so ``load_arm``'s message stays the
#: "write these three functions" one; enforced at run start instead, where the
#: consequence (a false ``loss_preregistration.json``) actually happens.
RUN_REQUIRED_HOOKS: tuple[str, ...] = ("loss_preregistration",)


def is_new_arm(arm: str) -> bool:
    return arm in ARM_MODULES


def arm_criteria(arm: str) -> tuple[str, ...]:
    return ARM_CRITERIA.get(arm, ())


# --------------------------------------------------------------------------- #
# what a head is handed
# --------------------------------------------------------------------------- #
@dataclass
class ArmContext:
    """Everything ``AmortModel.forward_geo`` has, handed to the arm in one object.

    ``h_cond`` is the readout of :mod:`q3vl.whereb.readout` -- ``(K, 2560)``,
    K = 1 for every single-position kind.  It is ``None`` only when the batch
    builder was not given a ``ReadoutBuilder``, which for these arms is a
    configuration error the head should refuse rather than work around (the
    pooled ``cond`` is the pathway S13 measured losing the geometry).

    ``h_where`` / ``h_mask`` are the full reply-span hiddens and their mask, kept
    so a head that wants the span (``where_span_pool`` is already folded into
    ``h_cond``, but e.g. a cross-attention readout would want the rows) does not
    have to reach around the context.
    """

    feat: torch.Tensor                      # (1, 1024, gh, gw)
    grid_h: int
    grid_w: int
    h_cond: torch.Tensor | None = None      # (K, 2560)
    cond: torch.Tensor | None = None        # (1, cond_dim) pooled CondEncoder
    extra: torch.Tensor | None = None       # sim / center / geom channels
    phi_dir: torch.Tensor | None = None
    sim: torch.Tensor | None = None
    center: torch.Tensor | None = None
    geom: Any = None
    guide_hi: torch.Tensor | None = None
    h_where: torch.Tensor | None = None     # (1, T, 2560)
    h_mask: torch.Tensor | None = None      # (1, T)
    sample: Any = None                      # AmortSampleInputs, when available
    meta: dict[str, Any] = field(default_factory=dict)

    def require_cond(self, arm: str) -> torch.Tensor:
        """``(K, 2560)`` or a message that names the missing flag."""
        if self.h_cond is None:
            raise ValueError(
                f"arm {arm} reads h_cond (the <seg_where>-position hidden) and "
                "the batch builder supplied none.  Pass "
                "readout=ReadoutBuilder(tokenizer, ReadoutSpec(...)) to "
                "AmortBatchBuilder -- run_amort_arm does this whenever --arm is "
                "one of the EPR-018..023 arms.")
        return self.h_cond

    def require_vector(self, arm: str) -> torch.Tensor:
        """``(2560,)`` for the single-vector readouts."""
        h = self.require_cond(arm)
        if h.shape[0] != 1:
            raise ValueError(
                f"arm {arm} asked for one condition vector but the readout "
                f"produced {h.shape[0]}; say how the head aggregates them "
                "(the K>1 aggregation is NOVEL in every proposal and is "
                "pre-registered per arm)")
        return h[0]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
_CACHE: dict[str, Any] = {}


def load_arm(arm: str):
    """Import the arm module, validate its hooks, cache it.

    The error messages are the whole point of this function: until the six head
    modules exist, ``--arm SEGSAM`` must fail with "write this file, with these
    three functions" rather than with a bare ``ModuleNotFoundError``.
    """
    if arm in _CACHE:
        return _CACHE[arm]
    path = ARM_MODULES.get(arm)
    if path is None:
        raise KeyError(f"{arm!r} is not one of the registered arms {NEW_ARMS}")
    try:
        mod = importlib.import_module(path)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != path:
            raise                              # a real missing dependency
        raise ImportError(
            f"arm {arm} is registered but its module {path} does not exist yet.\n"
            f"Create {path.replace('.', '/')}.py exporting:\n"
            f"    ARM = {arm!r}\n"
            f"    CRITERIA = {ARM_CRITERIA[arm]!r}\n"
            f"    def build_head(*, in_dim, text_dim, args=None, **kw) -> nn.Module\n"
            f"    def forward(model, head, ctx) -> dict   # must contain 'm_low'\n"
            f"    def compute_loss(model, out, x, weights) -> AmortLoss\n"
            f"See q3vl/whereb/amort/arms.py for the optional hooks."
        ) from exc
    missing = [h for h in REQUIRED_HOOKS if not callable(getattr(mod, h, None))]
    if missing:
        raise ImportError(
            f"arm module {path} is missing the required hook(s) {missing}; "
            f"see q3vl/whereb/amort/arms.py for the contract")
    name = getattr(mod, "ARM", None)
    if name != arm:
        raise ImportError(
            f"arm module {path} declares ARM={name!r} but is registered as "
            f"{arm!r}; the two must agree or the criterion table addresses the "
            "wrong module")
    crit = tuple(getattr(mod, "CRITERIA", ()) or ())
    if crit != ARM_CRITERIA[arm]:
        raise ImportError(
            f"arm module {path} declares CRITERIA={crit!r} but the registry "
            f"pre-registers {ARM_CRITERIA[arm]!r}; an arm may not rename its own "
            "pre-registered criterion column")
    _CACHE[arm] = mod
    return mod


def arm_hook(arm: str, name: str):
    """An optional hook, or ``None``.  Unknown hook names are a typo, not a
    silent no-op."""
    if name not in REQUIRED_HOOKS + OPTIONAL_HOOKS:
        raise KeyError(
            f"{name!r} is not part of the arm contract; known hooks: "
            f"{REQUIRED_HOOKS + OPTIONAL_HOOKS}")
    fn = getattr(load_arm(arm), name, None)
    return fn if callable(fn) else None
