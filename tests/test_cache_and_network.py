from __future__ import annotations

import json

from run import fingerprint, task_command, valid_cache
from scripts.generate_network import ensure_network, sha256_file


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_cache_requires_success_and_matching_fingerprint(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "ok", "fingerprint": "right"}))
    assert valid_cache(path, "right")
    assert not valid_cache(path, "wrong")
    path.write_text(json.dumps({"status": "failed", "fingerprint": "right"}))
    assert not valid_cache(path, "right")


def test_network_cache_is_deterministic_and_verified(tmp_path) -> None:
    first_path, first = ensure_network(tmp_path, 100, 10, 0.05, 42)
    second_path, second = ensure_network(tmp_path, 100, 10, 0.05, 42)
    assert first_path == second_path
    assert first == second
    assert first["edges"] == 500
    assert first["mean_degree_observed"] == 10
    assert first["sha256"] == sha256_file(first_path)


def test_task_command_passes_transmission_multiplier_to_each_runner() -> None:
    task = {
        "network": "network.tsv.gz",
        "network_sha256": "checksum",
        "network_edges": 50,
        "n": 10,
        "days": 5,
        "replicate": 1,
        "seed": 1,
        "mean_degree": 10.0,
        "target_r0": 2.0,
        "initial_infected": 1,
        "latent_days": 4.0,
        "infectious_days": 7.0,
        "hospitalization_probability": 0.05,
        "hospital_days": 7.0,
        "transmission_multiplier": 0.71,
        "fingerprint": "fingerprint",
        "engine_version": "version",
        "output": "result.json",
    }
    for engine in ("covasim", "epiworldR"):
        command = task_command(task | {"engine": engine})
        index = command.index("--transmission-multiplier")
        assert command[index + 1] == "0.71"
