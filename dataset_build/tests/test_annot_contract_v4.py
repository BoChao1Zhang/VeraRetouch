"""The v4 annotation contract: seven sections, geometry hints, leak boundary.

The band slot failed blind review more than any other (11/28) because its mask is
a strip that crosses the whole frame through the subject and the annotator was
never told so.  v4 fixes that on two sides: the prompt receives the candidate's
own geometry, and the output declares the edited region in a dedicated
``region_scope`` section.  The fixtures below are copied verbatim out of the
eval100 groups journal so the descriptors are exercised on real geometry.
"""
from __future__ import annotations

import contextlib
import io
import unittest
import unittest.mock

from construct import responses
from construct.responses import (
    ANNOTATION_FIELDS,
    ANNOTATION_JSON_SCHEMA,
    GLOBAL_REGION_SCOPE,
    PROMPT_VARIANT,
    REASONING_FIELDS,
    AnnotationError,
    _SECTION_TOKENS,
    assemble_reasoning,
    axis_bucket,
    build_prompt,
    compass_bucket,
    edit_geometry_hint,
    parse_annotation_json,
    split_reasoning,
    visual_angle,
)
from dataset_build.tools import reeval_annot_blind as blind
from dataset_build.tools import reeval_annot_mech as mech
from dataset_build.tools.reeval_annot_mech import check_sections, check_unit

# --- real candidate geometry, eval100-annotqa-20260727/groups.jsonl ----------
BAND_GEOMETRY = {
    "Angle": 86.65, "Bottom": 0.9364, "Feather": 70.0, "Flipped": "true",
    "Left": -1.3402, "Midpoint": 50.0, "Right": 1.8598, "Roundness": 0.0,
    "Top": 0.2187,
}
BAND_DIAGONAL_GEOMETRY = {
    "Angle": 54.21, "Bottom": 0.9381, "Feather": 55.0, "Flipped": "true",
    "Left": -1.3402, "Midpoint": 50.0, "Right": 1.8598, "Roundness": 0.0,
    "Top": 0.217,
}
RADIAL_GEOMETRY = {
    "Angle": 71.72, "Bottom": 0.8947, "Feather": 55.0, "Flipped": "true",
    "Left": -0.3017, "Midpoint": 50.0, "Right": 0.8213, "Roundness": 0.0,
    "Top": 0.2604,
}
LINEAR_GEOMETRY = {
    "Flipped": "false", "FullX": 0.6122, "FullY": 0.4581,
    "ZeroX": 0.8621, "ZeroY": 0.5419,
}

DIAGONAL_DOWN = "diagonal, running from the upper left down to the lower right"
DIAGONAL_UP = "diagonal, running from the lower left up to the upper right"

# Two candidates whose stored angle names one bucket in normalised coordinates and
# a different one on the picture the annotator is shown.  Both are real: the
# geometry and the before-image size are the ones eval100 actually holds, and they
# are the reason the orientation descriptor had to learn about the aspect ratio.
# 71.72 deg squashed into a 3:2 landscape frame reads as 63.6 deg -- a diagonal,
# not a vertical -- and 15.97 deg stretched into a 2:3 portrait frame reads as
# 23.2 deg, just past the horizontal edge.
LANDSCAPE_SIZE = (3648, 2432)
PORTRAIT_SIZE = (2048, 3072)
NEAR_VERTICAL_ON_LANDSCAPE = {**RADIAL_GEOMETRY, "Angle": 71.72}
NEAR_HORIZONTAL_ON_PORTRAIT = {**BAND_GEOMETRY, "Angle": 15.97}


def fields(**overrides: str) -> dict[str, str]:
    base = {
        "problem_lighting": "The light is flat across the picture.",
        "plan_lighting": "Lift the light on the standing figure.",
        "problem_global_color": "The overall palette is washed out.",
        "plan_global_color": "Recover a fuller overall palette.",
        "problem_specific_color": "The meadow greens read grey.",
        "plan_specific_color": "Bring the meadow greens back.",
        "region_scope": (
            "subject: the woman in the meadow; edit scope: a diagonal band through "
            "the subject, extending across the background from lower-left to "
            "upper-right"
        ),
        "instruction_long": "Please warm up the woman and the grass she is standing on.",
        "instruction_short": "Warm the woman and the grass.",
    }
    base.update(overrides)
    return base


def legacy_reasoning() -> str:
    """A six-section ``reasoning`` string in the shape every eval100 row has."""
    parts = fields()
    return (
        "<problem_light_start>" + parts["problem_lighting"] + "<problem_light_end>"
        "<problem_globalcolor_start>" + parts["problem_global_color"]
        + "<problem_globalcolor_end>"
        "<problem_specificcolor_start>" + parts["problem_specific_color"]
        + "<problem_specificcolor_end>"
        "<plan_light_start>" + parts["plan_lighting"] + "<plan_light_end>"
        "<plan_globalcolor_start>" + parts["plan_global_color"] + "<plan_globalcolor_end>"
        "<plan_specificcolor_start>" + parts["plan_specific_color"]
        + "<plan_specificcolor_end>"
    )


class SevenSectionContractTests(unittest.TestCase):
    def test_region_scope_is_a_required_field_between_problems_and_plans(self):
        self.assertIn("region_scope", ANNOTATION_FIELDS)
        text = assemble_reasoning(fields())
        self.assertLess(
            text.index("<problem_specificcolor_end>"), text.index("<region_scope_start>")
        )
        self.assertLess(text.index("<region_scope_end>"), text.index("<plan_light_start>"))
        self.assertEqual(len(split_reasoning(text)), 7)
        self.assertIn("diagonal band", split_reasoning(text)["region_scope"])

    def test_seven_field_response_parses_and_a_missing_section_is_schema_failed(self):
        import json

        parsed = parse_annotation_json(json.dumps(fields()))
        self.assertEqual(set(parsed), set(ANNOTATION_FIELDS))

        missing = fields()
        missing.pop("region_scope")
        with self.assertRaises(AnnotationError) as caught:
            parse_annotation_json(json.dumps(missing))
        self.assertEqual(caught.exception.code, "schema_failed")
        # A bad draw, not a terminal task: the annotator bounds the redraws.
        self.assertTrue(caught.exception.retryable)

        with self.assertRaises(AnnotationError) as short:
            parse_annotation_json(json.dumps(fields(region_scope="a b")))
        self.assertEqual(short.exception.code, "schema_failed")

    def test_field_order_is_the_section_emission_order(self):
        # A strict json_schema generates its properties in declaration order, so
        # this order is also the order the model writes them in: the three
        # problems, then the declaration of how far the edit reaches, then the
        # three plans that are conditioned on it, then the instructions.
        self.assertEqual(
            ANNOTATION_FIELDS,
            (
                "problem_lighting", "problem_global_color", "problem_specific_color",
                "region_scope",
                "plan_lighting", "plan_global_color", "plan_specific_color",
                "instruction_long", "instruction_short",
            ),
        )
        self.assertEqual(REASONING_FIELDS, ANNOTATION_FIELDS[:7])
        # the schema slice and the emitted reasoning are the same seven, in order
        self.assertEqual(tuple(_SECTION_TOKENS), REASONING_FIELDS)
        self.assertEqual(ANNOTATION_JSON_SCHEMA["required"], list(ANNOTATION_FIELDS))
        scope = ANNOTATION_FIELDS.index("region_scope")
        for name in ("problem_lighting", "problem_global_color", "problem_specific_color"):
            self.assertLess(ANNOTATION_FIELDS.index(name), scope)
        for name in ("plan_lighting", "plan_global_color", "plan_specific_color"):
            self.assertGreater(ANNOTATION_FIELDS.index(name), scope)

    def test_legacy_six_section_text_still_reads_back(self):
        sections = split_reasoning(legacy_reasoning())
        self.assertEqual(len(sections), 6)
        self.assertNotIn("region_scope", sections)
        self.assertEqual(sections["plan_lighting"], "Lift the light on the standing figure.")


class DirectionBucketTests(unittest.TestCase):
    def test_axis_buckets_and_their_boundaries(self):
        cases = {
            0.0: "horizontal", 22.4: "horizontal", 22.5: DIAGONAL_DOWN,
            44.9: DIAGONAL_DOWN, 45.0: DIAGONAL_DOWN, 67.4: DIAGONAL_DOWN,
            67.5: "vertical", 90.0: "vertical", 112.4: "vertical",
            112.5: DIAGONAL_UP, 135.0: DIAGONAL_UP, 157.4: DIAGONAL_UP,
            157.5: "horizontal", 180.0: "horizontal",
            # negative and wrapped angles land on the same undirected axis
            -71.72: "vertical", -45.0: DIAGONAL_UP, 266.65: "vertical",
        }
        for angle, expected in cases.items():
            with self.subTest(angle=angle):
                self.assertEqual(axis_bucket(angle), expected)

    def test_compass_buckets_use_image_coordinates(self):
        self.assertEqual(compass_bucket(1.0, 0.0), "right edge")
        self.assertEqual(compass_bucket(-1.0, 0.0), "left edge")
        self.assertEqual(compass_bucket(0.0, 1.0), "bottom edge")
        self.assertEqual(compass_bucket(0.0, -1.0), "top edge")
        self.assertEqual(compass_bucket(1.0, -1.0), "upper-right corner")
        self.assertEqual(compass_bucket(-1.0, 1.0), "lower-left corner")

    def test_visual_angle_reads_a_normalised_angle_off_the_displayed_picture(self):
        # 0 and 90 are the axes themselves, so no aspect ratio can move them; a
        # landscape frame flattens everything in between and a portrait frame
        # steepens it, by exactly atan(H/W * tan(theta)).
        cases = (
            (45.0, None, 45.0), (45.0, (1000, 1000), 45.0),
            (0.0, (3000, 1000), 0.0), (90.0, (3000, 1000), 90.0),
            (45.0, (3000, 1000), 18.43), (45.0, (1000, 3000), 71.57),
            (71.72, LANDSCAPE_SIZE, 63.64), (15.97, PORTRAIT_SIZE, 23.23),
            # a degenerate size is not trusted and falls back to the raw angle
            (45.0, (0, 1000), 45.0),
        )
        for angle, size, expected in cases:
            with self.subTest(angle=angle, size=size):
                self.assertAlmostEqual(visual_angle(angle, size), expected, places=2)


class GeometryHintTests(unittest.TestCase):
    def hint(self, slot_mode, geometry, variant, size=None):
        return edit_geometry_hint(
            {"slot_mode": slot_mode, "geometry": geometry}, variant=variant, size=size
        )

    def test_the_orientation_bucket_follows_the_aspect_ratio_of_the_before_image(self):
        # The two real counterexamples: read in normalised coordinates each hint
        # names a bucket the viewer cannot see.
        radial = self.hint("radial", NEAR_VERTICAL_ON_LANDSCAPE, "v4a", LANDSCAPE_SIZE)
        self.assertIn(f"its long axis {DIAGONAL_DOWN}", radial)
        self.assertNotIn("its long axis vertical", radial)
        self.assertIn(
            "its long axis vertical",
            self.hint("radial", NEAR_VERTICAL_ON_LANDSCAPE, "v4a"),
        )

        band = self.hint("band", NEAR_HORIZONTAL_ON_PORTRAIT, "v4a", PORTRAIT_SIZE)
        self.assertIn(f"straight band, {DIAGONAL_DOWN}", band)
        self.assertNotIn("straight band, horizontal", band)
        self.assertIn(
            "straight band, horizontal",
            self.hint("band", NEAR_HORIZONTAL_ON_PORTRAIT, "v4a"),
        )

    def test_a_square_frame_leaves_every_descriptor_exactly_as_it_was(self):
        # The regression guard: normalised and visual coordinates coincide at 1:1,
        # so a square before image must produce byte-identical text to no size at
        # all -- which is also the fallback when the size cannot be read.
        cases = (
            ("band", BAND_GEOMETRY), ("band", BAND_DIAGONAL_GEOMETRY),
            ("band", NEAR_HORIZONTAL_ON_PORTRAIT), ("radial", RADIAL_GEOMETRY),
            ("radial", NEAR_VERTICAL_ON_LANDSCAPE), ("linear", LINEAR_GEOMETRY),
        )
        for slot_mode, geometry in cases:
            for variant in ("v4a", "v4b"):
                with self.subTest(slot_mode=slot_mode, variant=variant,
                                  angle=geometry.get("Angle")):
                    self.assertEqual(
                        self.hint(slot_mode, geometry, variant, (2000, 2000)),
                        self.hint(slot_mode, geometry, variant),
                    )

    def test_linear_axis_and_compass_are_read_off_the_displayed_picture(self):
        # The same gradient vector points at the left edge of a wide frame and at
        # the upper-left corner of a tall one.
        wide = self.hint("linear", LINEAR_GEOMETRY, "v4a", (4000, 1000))
        self.assertIn("horizontal linear gradient", wide)
        self.assertIn("strongest toward the left edge", wide)

        tall = self.hint("linear", LINEAR_GEOMETRY, "v4a", (1000, 4000))
        self.assertIn(f"a {DIAGONAL_DOWN} linear gradient", tall)
        self.assertIn("strongest toward the upper-left corner", tall)

    def test_v4a_band_states_the_orientation_and_the_reach_past_the_subject(self):
        text = self.hint("band", BAND_GEOMETRY, "v4a")
        self.assertIn("broad straight band, vertical", text)
        self.assertIn("runs off both edges of the frame", text)
        self.assertIn("covers background on both sides of the subject", text)
        self.assertIn("the subject is not the extent of the edit", text)
        # No number from the geometry may reach the model in v4a.
        for token in ("86.65", "0.9364", "1.8598", "70"):
            self.assertNotIn(token, text)

        diagonal = self.hint("band", BAND_DIAGONAL_GEOMETRY, "v4a")
        self.assertIn(DIAGONAL_DOWN, diagonal)

    def test_v4a_band_width_buckets(self):
        # Edges at 0.35 and 0.60 of the frame.  At the old 0.25/0.50 only 7 of the
        # 140 eval100 bands were ever "narrow" and the rest piled into "broad", so
        # the descriptor carried almost no information.
        for half_width, expected in ((0.10, "narrow"), (0.17, "narrow"),
                                     (0.175, "moderately wide"),
                                     (0.20, "moderately wide"),
                                     (0.29, "moderately wide"), (0.30, "broad"),
                                     (0.36, "broad")):
            geometry = {**BAND_GEOMETRY, "Top": 0.5 - half_width, "Bottom": 0.5 + half_width}
            with self.subTest(half_width=half_width):
                self.assertIn(f"a {expected} straight band", self.hint("band", geometry, "v4a"))

    def test_v4a_radial_and_linear_declare_the_background_reach(self):
        radial = self.hint("radial", RADIAL_GEOMETRY, "v4a")
        self.assertIn("large oval falloff centred on the subject", radial)
        self.assertIn("its long axis vertical", radial)
        self.assertIn("the nearby background changes too", radial)

        linear = self.hint("linear", LINEAR_GEOMETRY, "v4a")
        self.assertIn("horizontal linear gradient spanning the whole frame", linear)
        self.assertIn("strongest toward the left edge", linear)
        self.assertIn("as much as the subject does", linear)

    def test_flipped_inverts_which_side_the_edit_lives_on(self):
        outside = self.hint("band", {**BAND_GEOMETRY, "Flipped": "false"}, "v4a")
        self.assertIn("everything outside that band which changes", outside)

        flipped_linear = self.hint("linear", {**LINEAR_GEOMETRY, "Flipped": "true"}, "v4a")
        self.assertIn("strongest toward the right edge", flipped_linear)

    def test_v4b_hands_over_the_raw_numbers(self):
        band = self.hint("band", BAND_GEOMETRY, "v4b")
        self.assertIn("shape=band", band)
        self.assertIn("axis_angle=86.65 degrees", band)
        self.assertIn("center=(0.260, 0.578)", band)
        self.assertIn("half_axis_along=1.600", band)
        self.assertIn("half_axis_across=0.359", band)
        self.assertIn("feather=70", band)
        self.assertIn("applies inside this shape", band)
        self.assertIn("reaches past the subject", band)

        self.assertIn("shape=ellipse", self.hint("radial", RADIAL_GEOMETRY, "v4b"))
        linear = self.hint("linear", LINEAR_GEOMETRY, "v4b")
        self.assertIn("zero_end=(0.862, 0.542)", linear)
        self.assertIn("full_end=(0.612, 0.458)", linear)
        self.assertIn("flipped=false", linear)

    def test_v4b_states_which_frame_each_kind_of_number_lives_in(self):
        # Positions and extents stay normalised, because a fraction of the frame
        # means the same thing at any aspect ratio.  An angle does not, so
        # axis_angle is converted and the preamble says which is which.
        band = self.hint("band", BAND_GEOMETRY, "v4b", LANDSCAPE_SIZE)
        self.assertIn("a fraction of the frame's own width or height", band)
        self.assertIn("measured on the picture as you actually see it", band)
        self.assertIn("axis_angle=84.98 degrees", band)
        # positions and extents are untouched by the aspect ratio
        self.assertIn("center=(0.260, 0.578)", band)
        self.assertIn("half_axis_across=0.359", band)
        self.assertIn(
            "axis_angle=87.77 degrees",
            self.hint("band", BAND_GEOMETRY, "v4b", (2432, 3648)),
        )
        # a linear gradient states no angle, so it carries no angle convention
        linear = self.hint("linear", LINEAR_GEOMETRY, "v4b", LANDSCAPE_SIZE)
        self.assertIn("a fraction of the frame's own width or height", linear)
        self.assertNotIn("axis_angle", linear)

    def test_v4b_linear_strength_wording_follows_flipped(self):
        # Canonical rows are all flipped=false, so this is a latent trap rather
        # than a live bug: the wording must not contradict the flag next to it.
        upright = self.hint("linear", LINEAR_GEOMETRY, "v4b")
        self.assertIn("flipped=false", upright)
        self.assertIn("none at the zero end to full at the full end", upright)

        flipped = self.hint("linear", {**LINEAR_GEOMETRY, "Flipped": "true"}, "v4b")
        self.assertIn("flipped=true", flipped)
        self.assertIn("full at the zero end to none at the full end", flipped)
        self.assertNotIn("none at the zero end to full at the full end", flipped)

    def test_legacy_plus_prefixed_string_geometry_is_understood(self):
        stringy = {key: (f"+{value}" if isinstance(value, (int, float)) and value >= 0
                         else str(value)) for key, value in BAND_GEOMETRY.items()}
        self.assertIn("straight band, vertical", self.hint("band", stringy, "v4a"))
        self.assertIn("axis_angle=86.65 degrees", self.hint("band", stringy, "v4b"))

    def test_modes_without_geometry_get_no_hint(self):
        self.assertIsNone(edit_geometry_hint({"slot_mode": "semantic", "geometry": None}))
        self.assertIsNone(edit_geometry_hint({"slot_mode": "linear_bisect", "geometry": {}}))
        self.assertIsNone(edit_geometry_hint({"slot_mode": "band", "geometry": None}))
        self.assertIsNone(edit_geometry_hint({}))

    def test_both_variants_are_reachable_and_the_default_is_the_module_constant(self):
        self.assertIn(PROMPT_VARIANT, {"v4a", "v4b"})
        candidate = {"slot_mode": "band", "geometry": BAND_GEOMETRY}
        default = edit_geometry_hint(candidate)
        self.assertEqual(default, edit_geometry_hint(candidate, variant=PROMPT_VARIANT))
        self.assertNotEqual(
            edit_geometry_hint(candidate, variant="v4a"),
            edit_geometry_hint(candidate, variant="v4b"),
        )


class PromptWiringTests(unittest.TestCase):
    def task(self, mode, **candidate):
        base = {
            "objective_hints": {
                "brightness": "brighter", "warmth": "warmer",
                "chroma": "richer", "contrast": "higher",
            },
            "region": "center", "subject": {"name": "person"},
            "style_name": "Style Name",
        }
        base.update(candidate)
        return {"group": {"render_mode": mode, "subject": {"name": "person"}},
                "candidate": base}

    def test_band_prompt_carries_the_geometry_hint(self):
        prompt = build_prompt(
            self.task("local", slot_mode="band", geometry=BAND_GEOMETRY)
        )
        self.assertIn("Edit-region geometry", prompt)
        self.assertIn("runs off both edges of the frame", prompt)
        self.assertIn("region_scope must identify the subject", prompt)

    def test_semantic_and_global_prompts_carry_no_geometry_hint(self):
        semantic = build_prompt(self.task("local", slot_mode="semantic", geometry=None))
        self.assertNotIn("Edit-region geometry", semantic)
        self.assertIn("edit stays", semantic)

        style = build_prompt(self.task("global"))
        self.assertNotIn("Edit-region geometry", style)
        self.assertIn(GLOBAL_REGION_SCOPE, style)


class MechanicalCheckerTests(unittest.TestCase):
    def unit(self, *, contract="auto", task_type="local", **overrides):
        parts = fields(**overrides)
        reasoning = (
            overrides["reasoning"] if "reasoning" in overrides
            else assemble_reasoning(parts)
        )
        return check_unit(
            "u", "sft", "current", task_type, parts["instruction_long"],
            parts["instruction_short"], reasoning, "Style Name", ["Style Name"],
            contract,
        )

    def test_seven_sections_pass_and_a_missing_one_is_flagged(self):
        clean = self.unit()
        self.assertEqual(clean["contract"], "v4")
        self.assertEqual(clean["flags"], {})

        broken = assemble_reasoning(fields()).replace(
            "<region_scope_start>", ""
        ).replace("<region_scope_end>", "")
        flagged = check_sections(broken, "v4")
        self.assertFalse(flagged["sections_present"])
        self.assertFalse(flagged["sections_balanced"])

    def test_legacy_six_section_text_is_recognised_not_failed(self):
        legacy = self.unit(reasoning=legacy_reasoning())
        self.assertEqual(legacy["contract"], "legacy")
        self.assertEqual(legacy["flags"], {})
        self.assertTrue(check_sections(legacy_reasoning())["sections_ordered"])
        # Forcing the v4 contract is what turns the same text into a failure.
        self.assertFalse(check_sections(legacy_reasoning(), "v4")["sections_present"])

    def test_shape_words_are_legal_in_region_scope_and_illegal_elsewhere(self):
        self.assertEqual(self.unit()["flags"], {})
        self.assertNotIn(
            "scope_leak_mask_geometry_leak",
            self.unit(region_scope=(
                "subject: the man; edit scope: a linear gradient with a radial "
                "falloff reaching past him into the background"
            ))["flags"],
        )

        in_instruction = self.unit(
            instruction_long="Warm the diagonal band across the meadow, please.",
        )
        self.assertIn("geometry_word_in_instruction", in_instruction["flags"])
        self.assertEqual(
            in_instruction["geometry_words"]["instruction"], ["band", "diagonal"]
        )

        in_plan = self.unit(plan_lighting="Lift the light along the vertical strip.")
        self.assertIn("geometry_word_in_reasoning", in_plan["flags"])

    def test_numeric_and_implementation_words_are_illegal_everywhere(self):
        angle = self.unit(plan_global_color="Rotate the warmth by 37.2 degrees.")
        self.assertIn("numeric_geometry_echo", angle["flags"])
        self.assertIn("leak_any", angle["flags"])

        in_scope = self.unit(region_scope=(
            "subject: the man; edit scope: a band at 37.2 degrees across the frame"
        ))
        self.assertIn("numeric_geometry_echo", in_scope["flags"])
        self.assertIn("scope_leak_numeric_unit", in_scope["flags"])

        feathered = self.unit(plan_specific_color="Feather the edge of the selection.")
        self.assertIn("leak_mask_geometry_leak", feathered["flags"])
        scope_feathered = self.unit(
            region_scope="subject: the man; edit scope: a band with a feathered edge"
        )
        self.assertIn("scope_leak_mask_geometry_leak", scope_feathered["flags"])

    def test_every_numeric_geometry_escape_is_caught(self):
        # ``\b`` does not fire on either side of an underscore, so every snake_case
        # key v4b hands over used to walk past all three detectors.  These are the
        # exact strings the adversarial review escaped with.
        escapes = (
            # snake_case keys, the whole point of the fix
            "axis_angle=86.65",
            "half_axis_along 1.6 and half_axis_across 0.36",
            "zero_end 0.862 0.542, full_end 0.612 0.458",
            "feather 70",
            # two bare decimals in one sentence
            "the band is 0.36 wide and reaches 1.6 along",
            "an ellipse from 0.12 to 0.93 vertically",
            # an explicit pair
            "the edit spans 0.26 by 0.58 of the picture",
            "the centre sits at x 0.26 and y 0.58",
            "centred at (0.260, 0.578)",
            # numbers spelled out, and percentages written out
            "a band tilted about thirty degrees from vertical",
            "a band tilted roughly seventy-two degrees",
            "the gradient starts at 86 percent of the way across",
            # the forms that already worked, kept as a regression
            "a band at 37.2 degrees across the frame",
            "the axis is 86.65 degrees",
            "the band covers 0.36 of the frame",
        )
        for text in escapes:
            with self.subTest(text=text):
                self.assertTrue(mech.NUMERIC_GEOMETRY_ECHO.search(text), text)
                self.assertIn(
                    "numeric_geometry_echo",
                    self.unit(region_scope=f"subject: the man; edit scope: {text}")["flags"],
                )

    def test_vague_fractions_and_ordinary_counting_are_not_numeric_echoes(self):
        # "a third of the frame" is deliberately let through: the same words carry
        # no measurement in "a third of the image is sky", and a rule that caught
        # one would condemn the other.
        for text in (
            "a strip about a third of the frame wide",
            "A third of the image is sky.",
            "The two figures on the bench sit in shadow.",
            "The building has three bright windows.",
            "Lift the shadows so the foreground reads clearly.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(mech.NUMERIC_GEOMETRY_ECHO.search(text), text)

    def test_shape_vocabulary_bypasses_are_caught_and_scene_description_is_not(self):
        pointing_at_the_edit = (
            "Warm the diagonal band across the meadow.",
            "Lift the light along the vertical strip.",
            "Lift the light along the bright stripe crossing the field.",
            "Brighten the ribbon of light through the scene.",
            "Warm the column of trees on one side.",
            "Warm the slice of field crossing the picture.",
            "Brighten a swathe running corner to corner.",
            "Lift the upper portion of the frame.",
            "Warm the top half and leave the rest.",
            "Brighten from one side to the other.",
            "Warm the left-hand side of the picture.",
            "Warm the leftward part of the scene.",
            "Brighten the area sweeping across the picture.",
            "Add a concentric brightening around him.",
            "Brighten radiating outward from the man.",
            "Lift the light on the north-west part of the field.",
            "Warm the region crossing the whole picture.",
        )
        # The other direction: the same vocabulary describing what is depicted.
        # "falls off" and "gradient" were dropped from the ban outright; the rest
        # are adjudicated word senses, exactly as the leak set already does.
        describing_the_picture = (
            "The light falls off toward the edges of the room.",
            "The sky has smooth gradients that should stay smooth.",
            "Recover the subtle tonal gradient in the sky.",
            "The horizontal lines of the building are pleasing.",
            "Keep the vertical lines of the columns straight.",
            "The camera orientation makes the scene feel cramped.",
            "The wheel axis is hidden in shadow.",
            "The oval table is too dark.",
            "The bands of cloud need more separation.",
            "The rock strips along the shore are muddy.",
        )
        for text in pointing_at_the_edit:
            with self.subTest(banned=text):
                self.assertTrue(mech.geometry_words(text), text)
                self.assertIn(
                    "geometry_word_in_reasoning", self.unit(plan_lighting=text)["flags"]
                )
        for text in describing_the_picture:
            with self.subTest(allowed=text):
                self.assertEqual(mech.geometry_words(text), [], text)
                self.assertEqual(self.unit(problem_lighting=text)["flags"], {})

    def test_word_caps_are_flagged_for_v4_and_counted_for_legacy(self):
        long_instruction = " ".join(["warm"] * 76) + "."
        long_short = " ".join(["warm"] * 31) + "."
        over = self.unit(
            instruction_long=long_instruction, instruction_short=long_short
        )
        self.assertIn("instruction_over_cap", over["flags"])
        self.assertIn("instruction_short_over_cap", over["flags"])
        self.assertEqual(over["short"]["words_long"], 76)

        legacy = self.unit(
            reasoning=legacy_reasoning(), instruction_long=long_instruction,
            instruction_short=long_short,
        )
        self.assertNotIn("instruction_over_cap", legacy["flags"])
        self.assertEqual(legacy["short"]["words_long"], 76)

        exact = self.unit(
            instruction_long=" ".join(["warm"] * 75),
            instruction_short=" ".join(["warm"] * 30),
        )
        self.assertNotIn("instruction_over_cap", exact["flags"])
        self.assertNotIn("instruction_short_over_cap", exact["flags"])

    def test_style_tasks_must_use_the_degenerate_scope(self):
        good = self.unit(
            task_type="style", region_scope=GLOBAL_REGION_SCOPE,
            instruction_long="Give this the Style Name look, warmer and richer.",
            instruction_short="Apply the Style Name look.",
        )
        self.assertNotIn("global_scope_not_degenerate", good["flags"])
        bad = self.unit(
            task_type="style",
            region_scope="subject: the whole picture; edit scope: everything",
            instruction_long="Give this the Style Name look, warmer and richer.",
            instruction_short="Apply the Style Name look.",
        )
        self.assertIn("global_scope_not_degenerate", bad["flags"])


class EvidenceDirectoryTests(unittest.TestCase):
    """``--out`` has no default, so a bare rerun cannot land on WP5's evidence."""

    def assert_out_is_required(self, main, argv):
        with contextlib.redirect_stderr(io.StringIO()) as captured:
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--out", captured.getvalue())

    def test_both_reeval_stages_refuse_to_run_without_an_output_directory(self):
        self.assert_out_is_required(mech.main, [])
        self.assert_out_is_required(blind.main, ["b1"])
        for module in (mech, blind):
            with self.subTest(module=module.__name__):
                self.assertFalse(hasattr(module, "DEFAULT_OUT"))


class RelayAuditTests(unittest.TestCase):
    def test_every_result_row_can_be_traced_to_its_rubric_and_schema(self):
        from dataset_build.tools.reeval_relay import prompt_digest

        content = [
            {"type": "input_text", "text": "rubric A"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"},
        ]
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        base = prompt_digest(content, schema)
        self.assertEqual(sorted(base), ["rubric_sha256", "schema_sha256"])

        # the images are pinned by the record key already, so they are not hashed
        self.assertEqual(
            prompt_digest(
                [content[0], {"type": "input_image", "image_url": "data:image/jpeg;base64,BB"}],
                schema,
            ),
            base,
        )
        # key order in the schema is not a difference
        self.assertEqual(
            prompt_digest(content, {"properties": schema["properties"], "type": "object"}),
            base,
        )
        # a changed rubric or a changed schema is
        self.assertNotEqual(
            prompt_digest([{"type": "input_text", "text": "rubric B"}], schema)["rubric_sha256"],
            base["rubric_sha256"],
        )
        self.assertNotEqual(
            prompt_digest(content, {**schema, "required": ["a"]})["schema_sha256"],
            base["schema_sha256"],
        )


class BlindRubricTests(unittest.TestCase):
    def test_region_scope_dimension_is_added_only_for_v4_text(self):
        v4 = {"reasoning": assemble_reasoning(fields())}
        old = {"reasoning": legacy_reasoning()}
        self.assertEqual(blind.dimensions_for(v4), blind.ALL_DIMENSIONS)
        self.assertEqual(blind.dimensions_for(old), blind.DIMENSIONS)
        # A mixed pair keeps the four-dimension B1 rubric, so b3 stays comparable.
        self.assertEqual(blind.dimensions_for(old, v4), blind.DIMENSIONS)

    def test_rubric_and_schema_follow_the_dimension_set(self):
        self.assertIn("region_scope -", blind.rules_for(blind.ALL_DIMENSIONS))
        self.assertNotIn("region_scope -", blind.rules_for(blind.DIMENSIONS))
        self.assertEqual(
            blind.single_schema(blind.ALL_DIMENSIONS)["properties"]["scores"]["required"],
            list(blind.ALL_DIMENSIONS),
        )
        self.assertEqual(
            blind.pair_schema(blind.DIMENSIONS)["properties"]["preference"]["required"],
            [*blind.DIMENSIONS, "overall"],
        )

    def test_pair_header_counts_the_dimensions_it_actually_asks_for(self):
        entry = {"before": None, "after": None, "cgt": None}
        texts = {"instruction": "Warm it up.", "instruction_short": "Warm it.",
                 "reasoning": legacy_reasoning()}
        v4_texts = {**texts, "reasoning": assemble_reasoning(fields())}
        with unittest.mock.patch.object(blind, "data_url", lambda _path: "data:,"):
            four, _ = blind.build_pair(entry, texts, texts)
            five, _ = blind.build_pair(entry, v4_texts, v4_texts)
        self.assertIn("all 4 dimensions", four[0]["text"])
        self.assertIn("all 5 dimensions", five[0]["text"])
        self.assertNotIn("all four dimensions", five[0]["text"])

    def test_validation_accepts_either_dimension_set_and_rejects_a_mixture(self):
        four = {dimension: 4 for dimension in blind.DIMENSIONS}
        five = {dimension: 4 for dimension in blind.ALL_DIMENSIONS}
        for scores in (four, five):
            blind.validate_single(
                {"scores": scores,
                 "reasons": {k: "because it does" for k in scores},
                 "verdict": "pass"}
            )
        with self.assertRaises(ValueError):
            blind.validate_single({"scores": {"consistency": 4}, "verdict": "pass"})
        with self.assertRaises(ValueError):
            blind.validate_pair({
                "scores_A": five, "scores_B": four,
                "preference": {**{k: "A" for k in blind.ALL_DIMENSIONS}, "overall": "A"},
                "reason": "the first one is better overall",
            })
        blind.validate_pair({
            "scores_A": five, "scores_B": five,
            "preference": {**{k: "A" for k in blind.ALL_DIMENSIONS}, "overall": "A"},
            "reason": "the first one is better overall",
        })


def v5_hints(**overrides: object) -> dict[str, object]:
    """A measured table whose every axis clears its dead band."""
    base: dict[str, object] = {
        "brightness": {"delta": -6.0, "direction": "darker"},
        "warmth": {"delta": 3.0, "direction": "warmer"},
        "chroma": {"delta": -8.0, "direction": "more muted"},
        "contrast": {"delta": 2.0, "direction": "higher contrast"},
        "hue_gm": {"delta": 7.0, "direction": "shifted toward magenta/red"},
        "surfaces": [],
    }
    base.update(overrides)
    return base


def plans(**overrides: str) -> dict[str, str]:
    """The five directive fields, empty unless a test fills one in."""
    base = {name: "" for name in (
        "plan_lighting", "plan_global_color", "plan_specific_color",
        "instruction_long", "instruction_short",
    )}
    base.update(overrides)
    return base


class AfterReferenceBanTests(unittest.TestCase):
    """A problem section describes the before image and may not point forward.

    37% of the v5 arm of the 88-sample panel did point forward, against 20% of
    the control, and the patterns below reproduce both rates on the archived
    text -- which is the evidence that they match the failure and not merely the
    word "after".
    """

    def test_every_family_the_panel_produced_is_caught(self):
        for text in (
            "The scene is flatter than in the finished image.",
            "The palette is warmer compared with the after version.",
            "The hair lacks the depth present afterward.",
            "It reads cooler than in the revised image.",
            "The edited result uses much deeper shadows.",
            "Colors feel less unified than the after image.",
            "The light is softer than the finished look.",
            "It is brighter than the second image.",
            "The tones fall short of the end result.",
            "There is less contrast than in the after.",
        ):
            with self.subTest(text=text):
                self.assertTrue(responses.after_reference_hits(text), text)

    def test_before_only_prose_and_intent_language_survive(self):
        for text in (
            "The scene is overly bright and airy, with restrained definition.",
            "The original palette is too warm and clean for the intended mood.",
            "The light lacks the atmosphere the 冷青橙暗哑 style calls for.",
            "The composition needs a more subdued, cinematic feeling.",
            "Detail in the couple and the rocky shore is already good.",
            "The desired mood is quieter than what the picture offers.",
            "Sunlight after the rain has left the pavement glaring.",
            "The subject is a sought-after landmark shot at midday.",
            "Shadow detail is thin and the highlights are close to clipping.",
        ):
            with self.subTest(text=text):
                self.assertEqual(responses.after_reference_hits(text), [], text)


class MachineVocabularyBanTests(unittest.TestCase):
    """The hint table is working data; none of its wording may reach the prose."""

    def test_pipeline_vocabulary_is_caught(self):
        for text in (
            "No individual color surface shows a clearly visible chroma change.",
            "Avoid any distinct color-surface shift across the frame.",
            "Give this connected scene area a subtly warmer appearance.",
            "Do not treat the woman as the full extent of the adjustment.",
            "Keep the atmosphere outside the affected area intact.",
            "Preserve the trees beyond the affected region.",
            "Warm the edited region without spilling into the sky.",
            "The change sits below the measurement threshold.",
            "This is a low-confidence reading of the greens.",
            "Follow the measured direction for the sky.",
            "The localized color change is modest here.",
            "The annotator should not invent a claim.",
        ):
            with self.subTest(text=text):
                self.assertTrue(responses.machine_vocab_hits(text), text)

    def test_ordinary_photographic_english_is_spared(self):
        # Each of these was checked against the 88-sample corpus: a blacklist
        # that eats them would burn redraws on answers that are already right.
        for text in (
            "Maintain the composition and the natural feel of the scene.",
            "Reproduce the 冷青橙暗哑 style across the entire photograph.",
            "Keep the water surface calm and the reflections readable.",
            "Warm the region around the cyclist and the flowering branches.",
            "Name the scene content that changes: the coat, the road, the sky.",
            "Keep saturation changes subtle across the seaside palette.",
            "The rock surfaces and the dark architectural surfaces stay legible.",
            "Lift the light on the standing figure without flattening it.",
            "Preserve the greens of the foliage and the blue of the water.",
            "The overall figure of the woman should stay recognisable.",
        ):
            with self.subTest(text=text):
                self.assertEqual(responses.machine_vocab_hits(text), [], text)


class DirectionClaimTests(unittest.TestCase):
    def test_each_axis_reports_the_direction_the_text_asks_for(self):
        for text, axis, direction in (
            ("Brighten the bird and the branch.", "brightness", "brighter"),
            ("Darken the sky over the ruin.", "brightness", "darker"),
            ("Give the scene a warmer feel.", "warmth", "warmer"),
            ("Cool the woman and the white clothing.", "warmth", "cooler"),
            ("Push the foliage into richer greens.", "chroma", "richer"),
            ("Make the rocks more muted and neutral.", "chroma", "more muted"),
            ("Add stronger contrast to the shoreline.", "contrast", "higher contrast"),
            ("Soften the contrast on the subject.", "contrast", "lower contrast"),
            ("Nudge the foliage toward green.", "hue_gm", "shifted toward green"),
            ("Push the cast more magenta.", "hue_gm", "shifted toward magenta/red"),
        ):
            with self.subTest(text=text):
                self.assertIn(direction, responses.asserted_directions(text).get(axis, ()))

    def test_a_promise_not_to_move_is_not_a_direction_claim(self):
        for text in (
            "Keep the foliage greens as rich as they are.",
            "Preserve the warmer skin tones already present.",
            "Retain the muted character of the background.",
            "Leave the sky unchanged rather than making it cooler.",
            "Avoid brightening the far corners.",
            "Do not darken the water beyond the railing.",
            "Reduce the heavy muted quality of the vegetation.",
        ):
            with self.subTest(text=text):
                self.assertEqual(responses.asserted_directions(text), {}, text)

    def test_a_comparative_naming_existing_content_is_not_a_claim(self):
        # Both false alarms the panel produced, and their directive twins.
        for text in (
            "Balance the warm sky with the cooler coastal shadows.",
            "Warm and enrich the muted monochrome appearance of the wall.",
            "Work with the darker foreground already in the frame.",
        ):
            with self.subTest(descriptive=text):
                claims = responses.asserted_directions(text)
                self.assertNotIn("cooler", claims.get("warmth", ()))
                self.assertNotIn("more muted", claims.get("chroma", ()))
                self.assertNotIn("darker", claims.get("brightness", ()))
        for text, axis, direction in (
            ("Take the shadows to a cooler cast.", "warmth", "cooler"),
            ("Bring the wall to a more muted beige.", "chroma", "more muted"),
        ):
            with self.subTest(directive=text):
                self.assertIn(direction, responses.asserted_directions(text)[axis])

    def test_tonal_senses_of_shared_words_are_not_colour_claims(self):
        for text in (
            "Give the scene richer shadows and more restrained highlights.",
            "Enrich the mood with deeper darks under the trees.",
            "Lighten the mood of the portrait without changing the palette.",
        ):
            with self.subTest(text=text):
                claims = responses.asserted_directions(text)
                self.assertNotIn("richer", claims.get("chroma", ()))
                self.assertNotIn("brighter", claims.get("brightness", ()))


class HintContradictionTests(unittest.TestCase):
    """The OVERALL lines are a veto; a plan that reverses one is a bad draw."""

    def test_each_axis_is_checked_against_its_measured_direction(self):
        for axis, text, asserted, measured in (
            ("brightness", "Brighten the bird and the surrounding woodland.",
             "brighter", "darker"),
            ("warmth", "Cool the woman and the stream around her.",
             "cooler", "warmer"),
            ("chroma", "Enrich the restrained colors of the wrestler's mask.",
             "richer", "more muted"),
            ("contrast", "Soften the contrast across the shoreline.",
             "lower contrast", "higher contrast"),
            ("hue_gm", "Nudge the whole frame toward green.",
             "shifted toward green", "shifted toward magenta/red"),
        ):
            with self.subTest(axis=axis):
                found = responses.hint_contradictions(
                    plans(plan_global_color=text), v5_hints()
                )
                self.assertEqual(
                    found,
                    [{"axis": axis, "asserted": asserted, "measured": measured}],
                )

    def test_agreeing_with_the_measured_direction_is_clean(self):
        for axis, text in (
            ("brightness", "Darken the bird and the surrounding woodland."),
            ("warmth", "Give the woman and the stream a warmer cast."),
            ("chroma", "Make the wrestler's mask more muted."),
            ("contrast", "Add stronger contrast across the shoreline."),
            ("hue_gm", "Push the whole frame more magenta."),
        ):
            with self.subTest(axis=axis):
                self.assertEqual(
                    responses.hint_contradictions(
                        plans(plan_global_color=text), v5_hints()
                    ),
                    [],
                )

    def test_an_axis_inside_its_dead_band_has_no_direction_to_reverse(self):
        # 0.4 chroma is below the 2.0 ROC operating point, so the table declines
        # to call it and "richer" is merely unlicensed, not contradicted.
        quiet = v5_hints(chroma={"delta": 0.4, "direction": "more muted"})
        self.assertEqual(
            responses.hint_contradictions(
                plans(plan_specific_color="Push the foliage into richer greens."),
                quiet,
            ),
            [],
        )

    def test_a_surface_that_moved_the_other_way_licenses_the_claim(self):
        # The low-confidence rule, read forwards: the table itself prints a
        # surface that disagrees with the whole-region figure, so a sentence
        # about that surface is legitimate.
        text = plans(plan_specific_color="Push the red coat into richer colour.")
        self.assertTrue(responses.hint_contradictions(text, v5_hints()))
        licensed = v5_hints(surfaces=[{
            "name": "red", "area": 0.12, "d_L": 1.0, "d_a": 4.0, "d_b": 1.0,
            "d_C": 7.5, "direction": "richer", "low_confidence": True,
        }])
        self.assertEqual(responses.hint_contradictions(text, licensed), [])

    def test_problem_sections_are_never_read_as_direction_claims(self):
        # "The palette is too warm" implies cooling; reading it as a claim that
        # the edit warmed anything would invert the whole check.
        fields = {
            "problem_global_color": "The palette is far too warm and much brighter "
                                    "than the mood needs, and the colours are richer "
                                    "than they should be.",
            **plans(),
        }
        self.assertEqual(responses.hint_contradictions(fields, v5_hints()), [])

    def test_a_missing_or_malformed_table_disables_the_check(self):
        text = plans(plan_global_color="Brighten the whole frame.")
        self.assertEqual(responses.hint_contradictions(text, None), [])
        self.assertEqual(responses.hint_contradictions(text, {}), [])
        self.assertEqual(
            responses.hint_contradictions(text, {"brightness": {"direction": "darker"}}),
            [],
        )


class ProseViolationTests(unittest.TestCase):
    def test_a_clean_annotation_reports_nothing(self):
        clean = {
            "problem_lighting": "The scene is overly bright and airy.",
            "problem_global_color": "The palette is clean and a little cold.",
            "problem_specific_color": "The foliage reads greyer than the mood wants.",
            "region_scope": GLOBAL_REGION_SCOPE,
            "plan_lighting": "Darken the seaside scene and hold the shadow detail.",
            "plan_global_color": "Give the whole frame a warmer, more muted cast.",
            "plan_specific_color": "Let the rocks and water settle into quieter colour.",
            "instruction_long": "Please darken this seaside photograph, warm it a "
                                "little and take the colour back so it feels calm.",
            "instruction_short": "Darken and warm the seaside scene with quieter colour.",
        }
        self.assertEqual(responses.prose_violations(clean, v5_hints()), [])

    def test_all_three_bans_are_reported_together(self):
        dirty = {
            "problem_lighting": "The scene is flatter than in the finished image.",
            "problem_global_color": "The palette is clean and cold.",
            "problem_specific_color": "No color surface shows a visible chroma change.",
            "region_scope": GLOBAL_REGION_SCOPE,
            "plan_lighting": "Brighten the seaside scene throughout.",
            "plan_global_color": "Warm the frame overall.",
            "plan_specific_color": "Keep the rocks as they are.",
            "instruction_long": "Please brighten and warm this seaside photograph.",
            "instruction_short": "Brighten and warm the seaside scene.",
        }
        reasons = responses.prose_violations(dirty, v5_hints())
        self.assertEqual(len(reasons), 3)
        self.assertIn("than in the finished", reasons[0])
        self.assertIn("chroma", reasons[1])
        self.assertIn("brightness", reasons[2])
        self.assertIn("darker", reasons[2])

    def test_the_code_is_a_bounded_redraw_that_does_not_rotate_lanes(self):
        self.assertEqual(
            responses._BAD_DRAW_LIMITS[responses.PROSE_VIOLATION],
            responses.PROSE_ATTEMPT_LIMIT,
        )
        self.assertEqual(responses.PROSE_ATTEMPT_LIMIT, 3)
        # Both production lanes serve the same model, so the text of a bad draw
        # says nothing about which lane produced it.
        self.assertNotIn(responses.PROSE_VIOLATION, responses._LANE_ROTATING_CODES)


class MonochromeNoteTests(unittest.TestCase):
    def hint_block(self, hints):
        return responses._objective_hints({"objective_hints": hints})

    def test_a_strong_desaturation_is_named_as_a_conversion(self):
        block = self.hint_block(v5_hints(
            chroma={"delta": -9.4, "direction": "more muted"}
        ))
        self.assertIn("monochrome or near-monochrome palette", block)
        self.assertIn("never write enrich", block)

    def test_a_moderate_or_contested_desaturation_gets_no_note(self):
        for name, hints in (
            ("moderate", v5_hints(chroma={"delta": -3.0, "direction": "more muted"})),
            ("richer", v5_hints(chroma={"delta": 9.0, "direction": "richer"})),
            ("surface disagrees", v5_hints(
                chroma={"delta": -9.4, "direction": "more muted"},
                surfaces=[{"name": "red", "area": 0.2, "d_L": 0.0, "d_a": 1.0,
                           "d_b": 1.0, "d_C": 8.0, "direction": "richer",
                           "low_confidence": True}],
            )),
        ):
            with self.subTest(name=name):
                self.assertNotIn("near-monochrome", self.hint_block(hints))


class SystemPromptClauseTests(unittest.TestCase):
    def test_the_v5_1_clauses_are_in_the_instructions(self):
        prompt = responses._SYSTEM_PROMPT
        for fragment in (
            "The three problem fields describe the first image and nothing else",
            "seen afterward",
            "Only promise to preserve, retain or maintain something you have been "
            "told did not change",
            "Report the strength you were given",
            "must not be written as gentle, slight, subtle or natural",
            "instruction_long must sound like one person asking another",
        ):
            with self.subTest(fragment=fragment[:40]):
                self.assertIn(fragment, prompt)

    def test_the_hint_block_forbids_reusing_its_own_wording(self):
        block = responses._objective_hints({"objective_hints": v5_hints()})
        self.assertIn("Never reuse the wording of this table", block)
        self.assertIn("Only promise to preserve, retain or maintain what this table "
                      "says did not move", block)

    def test_the_hint_block_no_longer_prints_the_words_the_gate_bans(self):
        # Every one of the five ``prose_violation`` redraws the WP16 smoke
        # triggered was this block's own lowercase running text coming straight
        # back out, and one draw evaded the ban by writing "colored surface".
        # A ban on wording the prompt keeps printing is a retry tax, so the
        # words were removed at source and this pins them out.
        for name, hints in (
            ("silent", v5_hints()),
            ("listed", v5_hints(surfaces=[{
                "name": "red", "area": 0.18, "d_L": 1.4, "d_a": -6.0,
                "d_b": -7.0, "d_C": -9.2, "direction": "more muted",
                "low_confidence": False}])),
            ("uncertain", v5_hints(chroma={"delta": 6.0, "direction": "richer"},
                                   surfaces=[{
                "name": "red", "area": 0.12, "d_L": 0.0, "d_a": -6.0,
                "d_b": -7.0, "d_C": -10.7, "direction": "more muted",
                "low_confidence": True}])),
        ):
            block = responses._objective_hints({"objective_hints": hints}).lower()
            for word in ("chroma", "colour surface", "color surface",
                         "colour-surface", "low confidence",
                         "computed from the pixels", "affected area"):
                with self.subTest(case=name, word=word):
                    self.assertNotIn(word, block)


class MechanicalProseFlagTests(unittest.TestCase):
    """The audit imports the gate's own patterns, so the two cannot drift."""

    def test_the_regexes_are_the_annotators_own_objects(self):
        self.assertIs(mech.AFTER_REFERENCE_RE, responses.AFTER_REFERENCE_RE)
        self.assertIs(mech.MACHINE_VOCAB_RE, responses.MACHINE_VOCAB_RE)

    def unit(self, hints=None, **overrides):
        parts = fields(**overrides)
        return check_unit(
            "u", "sft", "current", "style", parts["instruction_long"],
            parts["instruction_short"], assemble_reasoning(parts),
            "Style Name", [], "auto", hints,
        )

    def test_each_ban_becomes_its_own_flag(self):
        clean = self.unit(v5_hints(
            warmth={"delta": 3.0, "direction": "warmer"},
            brightness={"delta": 4.0, "direction": "brighter"},
        ))
        for flag in ("after_reference_in_problem", "machine_vocab_leak",
                     "hint_contradiction"):
            with self.subTest(flag=flag, case="clean"):
                self.assertNotIn(flag, clean["flags"])

        after = self.unit(
            problem_lighting="The finished image is far more severe."
        )
        self.assertIn("after_reference_in_problem", after["flags"])
        self.assertEqual(after["after_reference"], ["finished image"])

        vocab = self.unit(
            plan_specific_color="No color surface carries a chroma change."
        )
        self.assertIn("machine_vocab_leak", vocab["flags"])
        self.assertEqual(vocab["machine_vocab"], ["chroma", "color surface"])

        contra = self.unit(
            v5_hints(warmth={"delta": -3.0, "direction": "cooler"}),
        )
        self.assertIn("hint_contradiction", contra["flags"])
        self.assertEqual(
            contra["hint_contradiction"],
            [{"axis": "warmth", "asserted": "warmer", "measured": "cooler"}],
        )

    def test_a_corpus_checked_without_a_journal_scores_no_contradiction(self):
        without = self.unit(None)
        self.assertNotIn("hint_contradiction", without["flags"])
        self.assertEqual(without["hint_contradiction"], [])


if __name__ == "__main__":
    unittest.main()
