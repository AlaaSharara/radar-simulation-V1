"""Stable, independent deterministic RNG streams derived from a global seed and stream name.

Seeds are derived via SHA-256 rather than Python's process-randomized ``hash()``,
ensuring bit-for-bit reproducible stream seeds and draws across processes and
platforms.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from radarsim.config import MotionConfig


DEFAULT_STREAM_NAMES: tuple[str, ...] = (
    "path",
    "action_order",
    "speed",
    "body_shape",
    "fall_type",
    "scatterers",
    "rf_diffuse",
    "noise",
    "impairments",
)

_DOMAIN_TAG = "radarsim.rng.v1"


def _validate_global_seed(global_seed: int) -> int:
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError(
            f"global_seed must be an integer, got {global_seed!r} ({type(global_seed).__name__})"
        )
    return global_seed


def _validate_stream_name(stream_name: str) -> str:
    if not isinstance(stream_name, str) or not stream_name.strip():
        raise ValueError(f"stream_name must be a non-empty string, got {stream_name!r}")
    if "\0" in stream_name:
        raise ValueError("stream_name must not contain NUL characters")
    return stream_name


def derive_stream_seed(
    global_seed: int,
    stream_name: str,
    *,
    subkey: str | int | None = None,
) -> int:
    """Derive a stable 64-bit unsigned integer seed from global_seed and stream_name.

    Uses SHA-256 with NUL-delimited domain separation so distinct
    ``(global_seed, stream_name, subkey)`` tuples cannot collide through
    concatenation ambiguity.
    """
    seed_int = _validate_global_seed(global_seed)
    name = _validate_stream_name(stream_name)
    parts = [_DOMAIN_TAG, str(seed_int), name]
    if subkey is not None:
        if isinstance(subkey, bool):
            raise TypeError("subkey must be a string, integer, or None (not bool)")
        if isinstance(subkey, str):
            if not subkey.strip() or "\0" in subkey:
                raise ValueError("subkey string must be non-empty and contain no NUL characters")
            parts.append(f"s:{subkey}")
        elif isinstance(subkey, int):
            parts.append(f"i:{subkey}")
        else:
            raise TypeError(f"Unsupported subkey type: {type(subkey).__name__}")
    payload = "\0".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_generator(
    global_seed: int,
    stream_name: str,
    *,
    subkey: str | int | None = None,
) -> np.random.Generator:
    """Create a fresh NumPy PCG64 Generator for a derived stream."""
    derived = derive_stream_seed(global_seed, stream_name, subkey=subkey)
    return np.random.Generator(np.random.PCG64(derived))


class SeedManager:
    """Manages independent named RNG streams derived from a single global seed.

    Drawing from one stream never advances or perturbs any other stream.
    """

    def __init__(
        self,
        global_seed: int,
        stream_names: Iterable[str] | None = None,
        *,
        strict_streams: bool = False,
    ) -> None:
        self.global_seed: int = _validate_global_seed(global_seed)
        self.strict_streams: bool = bool(strict_streams)
        names = tuple(DEFAULT_STREAM_NAMES if stream_names is None else stream_names)
        if not names:
            raise ValueError("stream_names must contain at least one stream name")
        validated: list[str] = []
        seen: set[str] = set()
        for name in names:
            clean = _validate_stream_name(name)
            if clean in seen:
                raise ValueError(f"Duplicate stream name: {clean!r}")
            seen.add(clean)
            validated.append(clean)
        self._registered_streams: tuple[str, ...] = tuple(validated)
        self._generators: dict[str, np.random.Generator] = {}
        self._py_randoms: dict[str, random.Random] = {}

    @classmethod
    def from_motion_config(
        cls,
        motion_config: MotionConfig,
        *,
        override_seed: int | None = None,
        strict_streams: bool = False,
    ) -> SeedManager:
        seed = motion_config.scenario.random_seed if override_seed is None else override_seed
        return cls(
            global_seed=seed,
            stream_names=motion_config.seeding.independent_streams,
            strict_streams=strict_streams,
        )

    @property
    def registered_streams(self) -> tuple[str, ...]:
        return self._registered_streams

    def _check_stream_name(self, stream_name: str) -> str:
        clean = _validate_stream_name(stream_name)
        if self.strict_streams and clean not in self._registered_streams:
            raise KeyError(
                f"Unregistered RNG stream {clean!r}; registered streams: {list(self._registered_streams)}"
            )
        return clean

    def seed_for(self, stream_name: str, *, subkey: str | int | None = None) -> int:
        clean = self._check_stream_name(stream_name)
        return derive_stream_seed(self.global_seed, clean, subkey=subkey)

    def rng(self, stream_name: str) -> np.random.Generator:
        """Return the persistent NumPy Generator for ``stream_name``."""
        clean = self._check_stream_name(stream_name)
        if clean not in self._generators:
            self._generators[clean] = make_generator(self.global_seed, clean)
        return self._generators[clean]

    def spawn_rng(
        self, stream_name: str, *, subkey: str | int | None = None
    ) -> np.random.Generator:
        """Create a fresh, independent NumPy Generator for ``stream_name`` (and optional subkey)."""
        clean = self._check_stream_name(stream_name)
        return make_generator(self.global_seed, clean, subkey=subkey)

    def py_random(self, stream_name: str) -> random.Random:
        """Return a persistent stdlib ``random.Random`` instance for ``stream_name``."""
        clean = self._check_stream_name(stream_name)
        if clean not in self._py_randoms:
            self._py_randoms[clean] = random.Random(derive_stream_seed(self.global_seed, clean))
        return self._py_randoms[clean]

    def reset(self, stream_name: str | None = None) -> None:
        """Reset one stream or all instantiated streams back to their initial state."""
        if stream_name is None:
            self._generators.clear()
            self._py_randoms.clear()
            return
        clean = self._check_stream_name(stream_name)
        self._generators.pop(clean, None)
        self._py_randoms.pop(clean, None)

    def seed_map(self) -> dict[str, int]:
        """Return a deterministic mapping of all registered (and accessed) stream seeds."""
        names = list(self._registered_streams)
        for accessed in sorted(set(self._generators) | set(self._py_randoms)):
            if accessed not in names:
                names.append(accessed)
        return {name: derive_stream_seed(self.global_seed, name) for name in names}
