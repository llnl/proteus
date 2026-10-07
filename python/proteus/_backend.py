from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
from types import ModuleType
from typing import Iterable


_BACKEND_ENTRYPOINT_GROUP = "proteus.backends"
_BACKEND_KIND_ENV = "PROTEUS_BACKEND_KIND"
_BACKEND_VARIANT_ENV = "PROTEUS_BACKEND_VARIANT"
_LOCKED_BACKEND: tuple[str, str] | None = None


@dataclass(frozen=True)
class BackendPackageSpec:
    kind: str
    module_name: str
    priority: int


def _entry_points_for_group(group: str):
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    return list(discovered.get(group, []))


def _normalize_backend_spec(raw_spec: object) -> BackendPackageSpec:
    if not isinstance(raw_spec, dict):
        raise ImportError(
            "Proteus backend entry point must return a dict with backend metadata"
        )

    try:
        kind = str(raw_spec["kind"])
        module_name = str(raw_spec["module"])
        priority = int(raw_spec["priority"])
    except KeyError as exc:
        raise ImportError(
            f"Proteus backend metadata is missing required key {exc.args[0]!r}"
        ) from exc

    return BackendPackageSpec(kind=kind, module_name=module_name, priority=priority)


def _discover_backend_specs() -> list[BackendPackageSpec]:
    specs = []
    for ep in _entry_points_for_group(_BACKEND_ENTRYPOINT_GROUP):
        raw_spec = ep.load()()
        specs.append(_normalize_backend_spec(raw_spec))
    return specs


def _sorted_specs(specs: Iterable[BackendPackageSpec]) -> list[BackendPackageSpec]:
    return sorted(specs, key=lambda spec: (spec.priority, spec.kind), reverse=True)


def _backend_module(spec: BackendPackageSpec) -> ModuleType:
    return import_module(spec.module_name)


def available_backends() -> list[str]:
    return [spec.kind for spec in _sorted_specs(_discover_backend_specs())]


def available_backend_variants() -> list[str]:
    variants: list[str] = []
    for spec in _sorted_specs(_discover_backend_specs()):
        module = _backend_module(spec)
        variants.extend(str(item["id"]) for item in module.available_variant_specs())
    return variants


def _raise_no_backend() -> None:
    raise ImportError(
        "No Proteus backend is installed. Install one of "
        "proteus-python-backend-host-llvm22, "
        "proteus-python-backend-cuda12-llvm22, or "
        "proteus-python-backend-rocm72 from "
        "https://software.llnl.gov/proteus/wheels/simple/."
    )


def _spec_by_kind(kind: str) -> BackendPackageSpec:
    for spec in _sorted_specs(_discover_backend_specs()):
        if spec.kind == kind:
            return spec
    raise ImportError(
        f"Requested Proteus backend kind {kind!r} is not installed. "
        f"Available backends: {available_backends() or 'none'}"
    )


def _spec_for_variant(variant_id: str) -> BackendPackageSpec:
    for spec in _sorted_specs(_discover_backend_specs()):
        module = _backend_module(spec)
        if any(str(item["id"]) == variant_id for item in module.available_variant_specs()):
            return spec
    raise ImportError(
        f"Requested Proteus backend variant {variant_id!r} is not installed. "
        f"Available variants: {available_backend_variants() or 'none'}"
    )


def _selected_backend_spec() -> BackendPackageSpec:
    specs = _sorted_specs(_discover_backend_specs())
    if not specs:
        _raise_no_backend()

    variant_override = os.environ.get(_BACKEND_VARIANT_ENV, "").strip()
    if variant_override:
        return _spec_for_variant(variant_override)

    kind_override = os.environ.get(_BACKEND_KIND_ENV, "").strip()
    if kind_override:
        return _spec_by_kind(kind_override)

    compatible: list[BackendPackageSpec] = []
    host_spec: BackendPackageSpec | None = None
    for spec in specs:
        module = _backend_module(spec)
        if spec.kind == "host":
            host_spec = spec
        if module.is_runtime_compatible():
            compatible.append(spec)

    if compatible:
        return _sorted_specs(compatible)[0]
    if host_spec is not None:
        return host_spec
    return specs[0]


def _selected_variant_spec() -> dict[str, object]:
    spec = _selected_backend_spec()
    module = _backend_module(spec)
    variant_spec = module.active_variant_spec()
    if str(variant_spec["kind"]) != spec.kind:
        raise ImportError(
            f"Proteus backend {spec.kind!r} returned mismatched variant "
            f"kind {variant_spec['kind']!r}"
        )
    return variant_spec


def _ensure_process_lock() -> dict[str, object]:
    global _LOCKED_BACKEND

    variant = _selected_variant_spec()
    selection = (str(variant["kind"]), str(variant["id"]))
    if _LOCKED_BACKEND is None:
        _LOCKED_BACKEND = selection
        return variant
    if _LOCKED_BACKEND != selection:
        raise RuntimeError(
            "Proteus backend selection is locked for this process. "
            f"Already loaded {_LOCKED_BACKEND[0]!r}/{_LOCKED_BACKEND[1]!r}, "
            f"requested {selection[0]!r}/{selection[1]!r}."
        )
    return variant


def active_backend() -> str:
    return str(_selected_variant_spec()["kind"])


def active_backend_variant() -> str:
    return str(_selected_variant_spec()["id"])


def load_backend_module():
    variant = _ensure_process_lock()
    return _backend_module(_spec_by_kind(str(variant["kind"])))


def load_native_module():
    module = load_backend_module()
    return module.load_native_module()
