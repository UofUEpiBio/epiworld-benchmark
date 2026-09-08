#!/usr/bin/env python3
"""Generate and cache the shared Watts--Strogatz contact networks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import networkx as nx


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_network(
    cache_dir: Path,
    n: int,
    mean_degree: int,
    rewire_probability: float,
    seed: int,
) -> tuple[Path, dict]:
    """Return a deterministic cached edge list and its verified metadata."""
    if mean_degree <= 0 or mean_degree >= n or mean_degree % 2:
        raise ValueError("mean_degree must be a positive even integer below n")

    network_dir = cache_dir / "networks"
    stem = f"watts-strogatz_n{n}_k{mean_degree}_p{rewire_probability:g}_seed{seed}"
    edge_path = network_dir / f"{stem}.tsv.gz"
    metadata_path = network_dir / f"{stem}.json"
    expected = {
        "format_version": 1,
        "network_type": "watts_strogatz",
        "n": n,
        "mean_degree_requested": mean_degree,
        "rewire_probability": rewire_probability,
        "seed": seed,
    }

    if edge_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            matching = all(metadata.get(key) == value for key, value in expected.items())
            if matching and metadata.get("sha256") == sha256_file(edge_path):
                return edge_path, metadata
        except (OSError, ValueError):
            pass

    network_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.watts_strogatz_graph(n, mean_degree, rewire_probability, seed=seed)
    fd, temporary = tempfile.mkstemp(dir=network_dir, prefix=f".{edge_path.name}.")
    os.close(fd)
    try:
        # A fixed gzip timestamp makes the checksum reproducible as well as the graph.
        with open(temporary, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="ascii", newline="") as stream:
                    stream.write("source\ttarget\n")
                    for source, target in sorted(graph.edges()):
                        stream.write(f"{source}\t{target}\n")
        os.replace(temporary, edge_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    edge_count = graph.number_of_edges()
    metadata = expected | {
        "edges": edge_count,
        "mean_degree_observed": 2.0 * edge_count / n,
        "density": 2.0 * edge_count / (n * (n - 1)),
        "sha256": sha256_file(edge_path),
    }
    _atomic_json(metadata_path, metadata)
    return edge_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--mean-degree", type=int, required=True)
    parser.add_argument("--rewire-probability", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    path, metadata = ensure_network(
        args.cache_dir,
        args.n,
        args.mean_degree,
        args.rewire_probability,
        args.seed,
    )
    print(json.dumps({"path": str(path), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
