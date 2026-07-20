"""Strict TOML configuration for the canonical databuild pipeline."""
from __future__ import annotations

import dataclasses
import os
import re
import stat
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ImportError:  # Python 3.10 production environments
    import tomli as tomllib


class ConfigError(ValueError):
    """Raised before production state is mutated when configuration is invalid."""


PRESET_FORMATS = frozenset({"xmp", "lrtemplate", "lut"})
PRESET_FILTERS = frozenset({*PRESET_FORMATS, "all"})


@dataclass(frozen=True, slots=True)
class MixConfig:
    local: float
    global_: float


@dataclass(frozen=True, slots=True)
class SourcesConfig:
    subject_cache: Path
    postgres_dsn: str


@dataclass(frozen=True, slots=True)
class PresetsConfig:
    bank_dir: Path
    taxonomy: Path
    fidelity_de_max: float
    disabled_formats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderConfig:
    short_edge: int
    jpeg_quality: int
    gpu_concurrency: int
    diff_short_edge: int
    visible_de_min: float
    visible_fraction_de: float
    visible_fraction_min: float


@dataclass(frozen=True, slots=True)
class MasksConfig:
    linear_target_alpha_mass: float
    sam3_relabel_attempts: int


@dataclass(frozen=True, slots=True)
class ExternalEndpointConfig:
    id: str
    base_url: str
    api_key: str
    concurrency: int


@dataclass(frozen=True, slots=True)
class LocalAnnotationConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    enable_thinking: bool
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class AnnotationConfig:
    external_model: str
    image_long_edge: int
    image_jpeg_quality: int
    external_reasoning_effort: str
    external_max_output_tokens: int
    transport_attempts_per_round: int
    queue_rounds: int
    external_endpoints: tuple[ExternalEndpointConfig, ...]
    local: LocalAnnotationConfig


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    postgres_dsn: str


@dataclass(frozen=True, slots=True)
class DatabuildConfig:
    schema_version: int
    build_id: str
    seed: int
    target_groups: int
    output_root: Path
    preset_filter: str
    mix: MixConfig
    sources: SourcesConfig
    presets: PresetsConfig
    render: RenderConfig
    masks: MasksConfig
    annotation: AnnotationConfig
    viewer: ViewerConfig

    @property
    def requested_formats(self) -> frozenset[str]:
        if self.preset_filter == "all":
            return PRESET_FORMATS
        return frozenset({self.preset_filter})

    @property
    def effective_formats(self) -> frozenset[str]:
        return self.requested_formats.difference(self.presets.disabled_formats)

    @property
    def secrets(self) -> tuple[str, ...]:
        values = [e.api_key for e in self.annotation.external_endpoints]
        values.append(self.annotation.local.api_key)
        for url in (
            *(endpoint.base_url for endpoint in self.annotation.external_endpoints),
            self.annotation.local.base_url,
        ):
            values.append(url)
            values.extend(uri_secrets(url))
        for dsn in (self.sources.postgres_dsn, self.viewer.postgres_dsn):
            values.append(dsn)
            values.extend(uri_secrets(dsn))
        return tuple(v for v in values if v)

    def sanitized_dict(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw["output_root"] = str(self.output_root)
        raw["sources"]["subject_cache"] = str(self.sources.subject_cache)
        raw["sources"]["postgres_dsn"] = redact_uri(self.sources.postgres_dsn)
        raw["presets"]["bank_dir"] = str(self.presets.bank_dir)
        raw["presets"]["taxonomy"] = str(self.presets.taxonomy)
        raw["presets"]["disabled_formats"] = list(self.presets.disabled_formats)
        raw["mix"]["global"] = raw["mix"].pop("global_")
        raw["annotation"]["external_endpoints"] = [
            {
                **endpoint,
                "base_url": redact_uri(str(endpoint["base_url"])),
                "api_key": "<redacted>",
            }
            for endpoint in raw["annotation"]["external_endpoints"]
        ]
        raw["annotation"]["local"]["base_url"] = redact_uri(
            self.annotation.local.base_url
        )
        raw["annotation"]["local"]["api_key"] = "<redacted>"
        raw["viewer"]["postgres_dsn"] = redact_uri(self.viewer.postgres_dsn)
        raw["disabled_formats"] = list(self.presets.disabled_formats)
        raw["effective_formats"] = sorted(self.effective_formats)
        return raw


_CREDENTIAL_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|credential|"
    r"password|passwd|secret|token)$",
    re.IGNORECASE,
)
_URI_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|authorization|bearer|credential|"
    r"password|passwd|secret|token)\b\s*[:=]\s*)([^\s,;]+)"
)


def uri_secrets(value: str) -> tuple[str, ...]:
    """Extract credential values so standalone exception fragments are redacted too."""
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return ()
    values = [parsed.username or "", parsed.password or ""]
    values.extend(
        item for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if _CREDENTIAL_KEY.search(key)
    )
    return tuple(item for item in values if item)


def redact_uri(value: str) -> str:
    """Redact URI userinfo and credential-bearing query parameters."""
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return "<redacted-uri>"
    if not parsed.scheme or not parsed.netloc:
        return "<redacted>" if _CREDENTIAL_KEY.search(value) else value
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    if parsed.username is not None or parsed.password is not None:
        host = f"<redacted>@{host}"
    query = []
    for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        query.append((key, "<redacted>" if _CREDENTIAL_KEY.search(key) else item))
    return urllib.parse.urlunsplit(
        (parsed.scheme, host, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def redact_text(value: object, secrets: tuple[str, ...] = ()) -> str:
    """Sanitize exception/log text without relying on one field name."""
    text = str(value)
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    text = _URI_IN_TEXT.sub(lambda match: redact_uri(match.group(0)), text)
    return _ASSIGNMENT_SECRET.sub(r"\1<redacted>", text)


def _table(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing or invalid [{name}] table")
    return value


def _keys(table: Mapping[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = sorted(set(table).difference(allowed))
    missing = sorted(required.difference(table))
    if unknown:
        raise ConfigError(f"unknown field(s) in {where}: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"missing field(s) in {where}: {', '.join(missing)}")


def _typed(table: Mapping[str, Any], name: str, typ: type, where: str) -> Any:
    value = table[name]
    if typ is int and isinstance(value, bool):
        raise ConfigError(f"{where}.{name} must be an integer")
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}.{name} must be a number")
        return float(value)
    if not isinstance(value, typ):
        raise ConfigError(f"{where}.{name} must be {typ.__name__}")
    return value


def _nonempty(value: str, name: str) -> str:
    if not value.strip():
        raise ConfigError(f"{name} must not be empty")
    return value


def _absolute(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConfigError(f"{name} must be an absolute path")
    return path


def _url(value: str, name: str, schemes: frozenset[str]) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError as exc:
        raise ConfigError(f"{name} is not a valid URL") from exc
    if parsed.scheme not in schemes or not parsed.hostname:
        raise ConfigError(f"{name} must use one of: {', '.join(sorted(schemes))}")
    return value


def _positive(value: int | float, name: str, *, allow_zero: bool = False) -> None:
    if value < 0 if allow_zero else value <= 0:
        raise ConfigError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")


def _check_private_file(path: Path) -> None:
    try:
        info = path.stat()
    except OSError as exc:
        raise ConfigError(f"cannot stat config file: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError("config path must be a regular file")
    if info.st_uid != os.getuid():
        raise ConfigError("config file must be owned by the current user")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise ConfigError("real config file permissions must be exactly 0600")


def _validate_runtime_paths(config: DatabuildConfig) -> None:
    required_dirs = {
        "sources.subject_cache": config.sources.subject_cache,
        "presets.bank_dir": config.presets.bank_dir,
    }
    required_files = {"presets.taxonomy": config.presets.taxonomy}
    for name, path in required_dirs.items():
        if not path.is_dir():
            raise ConfigError(f"{name} is not an existing directory: {path}")
    for name, path in required_files.items():
        if not path.is_file():
            raise ConfigError(f"{name} is not an existing file: {path}")
    parent = config.output_root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ConfigError(f"output_root has no writable parent: {config.output_root}")


def _build(data: Mapping[str, Any]) -> DatabuildConfig:
    root_allowed = {
        "schema_version", "build_id", "seed", "target_groups", "output_root",
        "preset_filter", "mix", "sources", "presets", "render", "masks",
        "annotation", "viewer",
    }
    _keys(data, root_allowed, root_allowed, "root")
    schema_version = _typed(data, "schema_version", int, "root")
    if schema_version != 1:
        raise ConfigError("schema_version must equal 1")
    build_id = _nonempty(_typed(data, "build_id", str, "root"), "build_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", build_id):
        raise ConfigError("build_id contains unsupported characters")
    seed = _typed(data, "seed", int, "root")
    target_groups = _typed(data, "target_groups", int, "root")
    _positive(target_groups, "target_groups")
    output_root = _absolute(_typed(data, "output_root", str, "root"), "output_root")
    preset_filter = _typed(data, "preset_filter", str, "root").lower()
    if preset_filter not in PRESET_FILTERS:
        raise ConfigError("preset_filter must be xmp, lrtemplate, lut, or all")

    mix_t = _table(data, "mix")
    _keys(mix_t, {"local", "global"}, {"local", "global"}, "mix")
    local_ratio = _typed(mix_t, "local", float, "mix")
    global_ratio = _typed(mix_t, "global", float, "mix")
    if local_ratio < 0 or global_ratio < 0 or abs(local_ratio + global_ratio - 1.0) > 1e-9:
        raise ConfigError("mix.local and mix.global must be non-negative and sum to one")
    mix = MixConfig(local=local_ratio, global_=global_ratio)

    sources_t = _table(data, "sources")
    _keys(sources_t, {"subject_cache", "postgres_dsn"},
          {"subject_cache", "postgres_dsn"}, "sources")
    sources = SourcesConfig(
        subject_cache=_absolute(_typed(sources_t, "subject_cache", str, "sources"),
                                "sources.subject_cache"),
        postgres_dsn=_url(_typed(sources_t, "postgres_dsn", str, "sources"),
                          "sources.postgres_dsn", frozenset({"postgres", "postgresql"})),
    )

    presets_t = _table(data, "presets")
    preset_keys = {"bank_dir", "taxonomy", "fidelity_de_max", "disabled_formats"}
    _keys(presets_t, preset_keys, preset_keys, "presets")
    disabled_raw = _typed(presets_t, "disabled_formats", list, "presets")
    if any(not isinstance(item, str) for item in disabled_raw):
        raise ConfigError("presets.disabled_formats must contain strings")
    disabled = tuple(item.lower() for item in disabled_raw)
    if len(disabled) != len(set(disabled)):
        raise ConfigError("presets.disabled_formats must not contain duplicates")
    invalid_disabled = sorted(set(disabled).difference(PRESET_FORMATS))
    if invalid_disabled:
        raise ConfigError("invalid disabled preset format(s): " + ", ".join(invalid_disabled))
    fidelity = _typed(presets_t, "fidelity_de_max", float, "presets")
    _positive(fidelity, "presets.fidelity_de_max")
    presets = PresetsConfig(
        bank_dir=_absolute(_typed(presets_t, "bank_dir", str, "presets"),
                           "presets.bank_dir"),
        taxonomy=_absolute(_typed(presets_t, "taxonomy", str, "presets"),
                           "presets.taxonomy"),
        fidelity_de_max=fidelity,
        disabled_formats=disabled,
    )
    if preset_filter != "all" and preset_filter in disabled:
        raise ConfigError(f"requested preset_filter {preset_filter!r} is disabled")
    if not frozenset(PRESET_FORMATS).difference(disabled):
        raise ConfigError("presets.disabled_formats removes every renderer format")

    render_t = _table(data, "render")
    render_keys = {
        "short_edge", "jpeg_quality", "gpu_concurrency", "diff_short_edge",
        "visible_de_min", "visible_fraction_de", "visible_fraction_min",
    }
    _keys(render_t, render_keys, render_keys, "render")
    render = RenderConfig(
        short_edge=_typed(render_t, "short_edge", int, "render"),
        jpeg_quality=_typed(render_t, "jpeg_quality", int, "render"),
        gpu_concurrency=_typed(render_t, "gpu_concurrency", int, "render"),
        diff_short_edge=_typed(render_t, "diff_short_edge", int, "render"),
        visible_de_min=_typed(render_t, "visible_de_min", float, "render"),
        visible_fraction_de=_typed(render_t, "visible_fraction_de", float, "render"),
        visible_fraction_min=_typed(render_t, "visible_fraction_min", float, "render"),
    )
    for name in ("short_edge", "jpeg_quality", "gpu_concurrency", "diff_short_edge"):
        _positive(getattr(render, name), f"render.{name}")
    if render.short_edge != 1024 or render.jpeg_quality != 95:
        raise ConfigError("canonical rendering is fixed at short edge 1024 and JPEG quality 95")
    if not 0.0 <= render.visible_fraction_min <= 1.0:
        raise ConfigError("render.visible_fraction_min must be in [0, 1]")
    _positive(render.visible_de_min, "render.visible_de_min")
    _positive(render.visible_fraction_de, "render.visible_fraction_de")

    masks_t = _table(data, "masks")
    mask_keys = {"linear_target_alpha_mass", "sam3_relabel_attempts"}
    _keys(masks_t, mask_keys, mask_keys, "masks")
    masks = MasksConfig(
        linear_target_alpha_mass=_typed(
            masks_t, "linear_target_alpha_mass", float, "masks"),
        sam3_relabel_attempts=_typed(masks_t, "sam3_relabel_attempts", int, "masks"),
    )
    if abs(masks.linear_target_alpha_mass - 0.5) > 1e-9:
        raise ConfigError("masks.linear_target_alpha_mass is fixed at 0.50")
    if not 0 <= masks.sam3_relabel_attempts <= 2:
        raise ConfigError("masks.sam3_relabel_attempts must be in [0, 2]")

    annotation_t = _table(data, "annotation")
    annotation_keys = {
        "external_model", "image_long_edge", "image_jpeg_quality",
        "external_reasoning_effort", "external_max_output_tokens",
        "transport_attempts_per_round", "queue_rounds", "external_endpoints", "local",
    }
    _keys(annotation_t, annotation_keys, annotation_keys, "annotation")
    endpoint_rows = _typed(annotation_t, "external_endpoints", list, "annotation")
    if len(endpoint_rows) != 2:
        raise ConfigError("annotation.external_endpoints must contain exactly two endpoints")
    endpoints = []
    endpoint_keys = {"id", "base_url", "api_key", "concurrency"}
    for index, row in enumerate(endpoint_rows):
        if not isinstance(row, Mapping):
            raise ConfigError(f"annotation.external_endpoints[{index}] must be a table")
        where = f"annotation.external_endpoints[{index}]"
        _keys(row, endpoint_keys, endpoint_keys, where)
        endpoint = ExternalEndpointConfig(
            id=_nonempty(_typed(row, "id", str, where), f"{where}.id"),
            base_url=_url(_typed(row, "base_url", str, where), f"{where}.base_url",
                          frozenset({"http", "https"})),
            api_key=_nonempty(_typed(row, "api_key", str, where), f"{where}.api_key"),
            concurrency=_typed(row, "concurrency", int, where),
        )
        _positive(endpoint.concurrency, f"{where}.concurrency")
        endpoints.append(endpoint)
    if len({endpoint.id for endpoint in endpoints}) != 2:
        raise ConfigError("annotation external endpoint IDs must be distinct")

    local_t = _table(annotation_t, "local")
    local_keys = {
        "base_url", "api_key", "model", "temperature", "enable_thinking",
        "max_output_tokens",
    }
    _keys(local_t, local_keys, local_keys, "annotation.local")
    local_annotation = LocalAnnotationConfig(
        base_url=_url(_typed(local_t, "base_url", str, "annotation.local"),
                      "annotation.local.base_url", frozenset({"http", "https"})),
        api_key=_nonempty(_typed(local_t, "api_key", str, "annotation.local"),
                          "annotation.local.api_key"),
        model=_nonempty(_typed(local_t, "model", str, "annotation.local"),
                        "annotation.local.model"),
        temperature=_typed(local_t, "temperature", float, "annotation.local"),
        enable_thinking=_typed(local_t, "enable_thinking", bool, "annotation.local"),
        max_output_tokens=_typed(local_t, "max_output_tokens", int, "annotation.local"),
    )
    if local_annotation.model != "qwen3_5-35b-a3b":
        raise ConfigError("annotation.local.model must be qwen3_5-35b-a3b")
    if abs(local_annotation.temperature - 0.2) > 1e-9:
        raise ConfigError("annotation.local.temperature must equal 0.2")
    if local_annotation.enable_thinking:
        raise ConfigError("annotation.local.enable_thinking must be false")
    if local_annotation.max_output_tokens != 2048:
        raise ConfigError("annotation.local.max_output_tokens must equal 2048")

    annotation = AnnotationConfig(
        external_model=_nonempty(
            _typed(annotation_t, "external_model", str, "annotation"),
            "annotation.external_model"),
        image_long_edge=_typed(annotation_t, "image_long_edge", int, "annotation"),
        image_jpeg_quality=_typed(annotation_t, "image_jpeg_quality", int, "annotation"),
        external_reasoning_effort=_typed(
            annotation_t, "external_reasoning_effort", str, "annotation"),
        external_max_output_tokens=_typed(
            annotation_t, "external_max_output_tokens", int, "annotation"),
        transport_attempts_per_round=_typed(
            annotation_t, "transport_attempts_per_round", int, "annotation"),
        queue_rounds=_typed(annotation_t, "queue_rounds", int, "annotation"),
        external_endpoints=tuple(endpoints),
        local=local_annotation,
    )
    if annotation.image_long_edge != 768 or annotation.image_jpeg_quality != 90:
        raise ConfigError("annotation images are fixed at longest edge 768 and JPEG quality 90")
    if annotation.external_reasoning_effort != "medium":
        raise ConfigError("annotation.external_reasoning_effort must be medium")
    if annotation.external_max_output_tokens != 6000:
        raise ConfigError("annotation.external_max_output_tokens must equal 6000")
    if not 1 <= annotation.transport_attempts_per_round <= 4:
        raise ConfigError("annotation.transport_attempts_per_round must be in [1, 4]")
    if not 1 <= annotation.queue_rounds <= 3:
        raise ConfigError("annotation.queue_rounds must be in [1, 3]")

    viewer_t = _table(data, "viewer")
    _keys(viewer_t, {"postgres_dsn"}, {"postgres_dsn"}, "viewer")
    viewer = ViewerConfig(
        postgres_dsn=_url(_typed(viewer_t, "postgres_dsn", str, "viewer"),
                          "viewer.postgres_dsn", frozenset({"postgres", "postgresql"})))

    return DatabuildConfig(
        schema_version=schema_version,
        build_id=build_id,
        seed=seed,
        target_groups=target_groups,
        output_root=output_root,
        preset_filter=preset_filter,
        mix=mix,
        sources=sources,
        presets=presets,
        render=render,
        masks=masks,
        annotation=annotation,
        viewer=viewer,
    )


def load_config(
    path: str | os.PathLike[str],
    *,
    require_private: bool = True,
    validate_paths: bool = True,
) -> DatabuildConfig:
    """Load and validate a canonical config without consulting environment variables."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        raise ConfigError("--config must be an absolute path")
    if require_private:
        _check_private_file(config_path)
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError("invalid TOML syntax in config") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {config_path}") from exc
    config = _build(data)
    if validate_paths:
        _validate_runtime_paths(config)
    return config
