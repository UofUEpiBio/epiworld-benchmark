"""Hardware and software environment of a benchmark run.

Timings depend on the machine and on the toolchains, so every scenario's
results are stored next to a description of where they were produced. Nothing
here may abort a run: a field that cannot be read becomes ``None`` (JSON null).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

# Set for every runner subprocess. Recorded with the run so a reader can see
# that engines were kept single-threaded.
RUNNER_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    "RCPP_PARALLEL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
    # Starsim otherwise rebuilds matplotlib's font cache on import.
    "STARSIM_INSTALL_FONTS": "0",
}

# The Dockerfile persists these as ENV so they can be read at run time.
PIN_VARIABLES = ("EPIWORLD_SHA", "EPIWORLDR_SHA", "FRED_SHA", "INDIVIDUAL_SHA")


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _run(command: list[str], **kwargs: Any) -> str | None:
    """Output of a command (empty if it printed nothing), or None if it is missing or fails."""
    try:
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True, timeout=30, **kwargs
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip()


def _first_line(text: str | None) -> str | None:
    return text.splitlines()[0].strip() if text else None


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def _int(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def cpu_model() -> str | None:
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string") or None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("model name", "Model Name", "Hardware") and value.strip():
            return value.strip()
    # aarch64 kernels do not report a model name in /proc/cpuinfo.
    for line in (_run(["lscpu"]) or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "Model name" and value.strip():
            return value.strip()
    return None


def physical_cores() -> int | None:
    if sys.platform == "darwin":
        return _int(_sysctl("hw.physicalcpu"))
    cores = set()
    for topology in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
        core, package = _read(topology / "core_id"), _read(topology / "physical_package_id")
        if core is not None:
            cores.add((package, core.strip()))
    return len(cores) or None


def total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        return _int(_sysctl("hw.memsize"))
    match = re.search(r"^MemTotal:\s+(\d+) kB", _read("/proc/meminfo") or "", re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def parse_cpu_max(text: str | None) -> float | None:
    """CPUs allowed by a cgroup v2 `cpu.max` ("<quota|max> <period>"); None if unlimited."""
    parts = (text or "").split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    try:
        return int(parts[0]) / int(parts[1])
    except (ValueError, ZeroDivisionError):
        return None


def parse_memory_max(text: str | None) -> int | None:
    """Bytes allowed by a cgroup v2 `memory.max`; None if unlimited."""
    return _int(text.strip()) if text else None


def hardware() -> dict[str, Any]:
    return {
        "cpu_model": cpu_model(),
        "logical_cores": os.cpu_count(),
        "physical_cores": physical_cores(),
        "memory_bytes": total_memory_bytes(),
        # Limits on this process's cgroup; null when unlimited or not readable.
        "cgroup_cpu_limit": parse_cpu_max(_read("/sys/fs/cgroup/cpu.max")),
        "cgroup_memory_limit_bytes": parse_memory_max(_read("/sys/fs/cgroup/memory.max")),
    }


def operating_system() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "container": Path("/run/.containerenv").exists() or Path("/.dockerenv").exists(),
    }


def toolchains() -> dict[str, str | None]:
    compiler = os.environ.get("CXX") or "c++"
    return {
        "python": platform.python_version(),
        "r": _first_line(_run(["R", "--version"])),
        "julia": _first_line(_run(["julia", "--version"])),
        "rust": _first_line(_run(["rustc", "--version"])),
        "cxx": _first_line(_run([compiler, "--version"])),
        "quarto": _first_line(_run(["quarto", "--version"])),
    }


def makefile_ref(name: str) -> str | None:
    match = re.search(rf"^{name}\s*:=\s*([0-9a-f]{{40}})\s*$", _read(ROOT / "Makefile") or "",
                      re.MULTILINE)
    return match.group(1) if match else None


def pins() -> dict[str, str | None]:
    """Commits of the sources built from git.

    The container image records them as environment variables. Natively,
    `make setup` downloads epiworld into `.deps/<EPIWORLD_REF>` and FRED into
    `$FRED_HOME`, with the commit in `COMMIT`; epiworldR and individual are
    installed outside this repository, so their commits are unknown.
    """
    found = {name: os.environ.get(name) or None for name in PIN_VARIABLES}
    if found["EPIWORLD_SHA"] is None:
        ref = makefile_ref("EPIWORLD_REF")
        found["EPIWORLD_SHA"] = ref if ref and (ROOT / ".deps" / ref).is_dir() else None
    if found["FRED_SHA"] is None:
        fred_home = Path(os.environ.get("FRED_HOME", ROOT / ".deps" / "fred"))
        found["FRED_SHA"] = (_read(fred_home / "COMMIT") or "").strip() or None
    return found


def repository() -> dict[str, Any]:
    """Commit of this checkout. Results are rewritten by every run, so changes
    under results/ do not make the checkout dirty."""
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    commit = _run([*git, "rev-parse", "HEAD"])
    status = _run([*git, "status", "--porcelain", "--", ".", ":(exclude)results"])
    return {
        "commit": commit or None,
        "dirty": None if commit is None or status is None else bool(status),
    }


def environment_id(environment: dict[str, Any]) -> str:
    """SHA-256 of what determines performance: hardware, OS, image, toolchains,
    pins, and engine versions. Timestamps, run settings, and repository state
    are excluded so identical setups share an id."""
    identity = {
        key: environment.get(key)
        for key in ("hardware", "os", "container_image", "toolchains", "pins", "engine_versions")
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect_environment(
    versions: dict[str, str], workers: int, profile: str
) -> dict[str, Any]:
    """Describe this machine, image, toolchains, and run settings.

    `versions` are the engine versions of the invocation, from `engine_versions()`.
    """
    environment = {
        "hardware": hardware(),
        "os": operating_system(),
        "container_image": os.environ.get("BENCHMARK_IMAGE_ID") or None,
        "toolchains": toolchains(),
        "pins": pins(),
        "engine_versions": dict(versions),
        "repository": repository(),
        "run": {
            "profile": profile,
            "workers": workers,
            "environment_variables": dict(RUNNER_ENV),
        },
    }
    return {"environment_id": environment_id(environment)} | environment
