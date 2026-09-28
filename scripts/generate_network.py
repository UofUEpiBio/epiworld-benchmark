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
import urllib.request

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


def _download_verified(url: str, path: Path, expected_sha256: str) -> None:
    """Download an immutable source file once and verify its content hash."""
    if path.exists() and sha256_file(path) == expected_sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as response, open(temporary, "wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        observed = sha256_file(Path(temporary))
        if observed != expected_sha256:
            raise ValueError(
                f"source checksum mismatch for {url}: expected {expected_sha256}, got {observed}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _matrix_market_edges(path: Path, n: int):
    """Yield zero-based edges in the induced subgraph on the first `n` rows."""
    with path.open("rt", encoding="ascii") as stream:
        header = stream.readline().strip()
        if not header.startswith("%%MatrixMarket matrix coordinate"):
            raise ValueError(f"unsupported MatrixMarket header in {path}: {header}")
        line = stream.readline()
        while line.startswith("%"):
            line = stream.readline()
        rows, columns, _entries = (int(value) for value in line.split())
        if rows != columns or n > rows:
            raise ValueError(f"{path} is {rows}x{columns}, cannot select {n} people")
        for line in stream:
            values = line.split()
            if len(values) < 2:
                continue
            source, target = int(values[0]) - 1, int(values[1]) - 1
            if source < n and target < n and source != target:
                yield (source, target) if source < target else (target, source)


def ensure_geopops_network(cache_dir: Path, n: int, config: dict) -> tuple[Path, dict]:
    """Collapse pinned GeoPops layers into one undirected, unweighted graph.

    GeoPops exports separate upper-triangular MatrixMarket adjacency matrices
    for household, workplace, school, and group-quarters contacts. This keeps
    the first `n` matrix rows, takes each layer's induced subgraph, unions the
    edges, removes duplicates and self edges, and writes the common edge-list
    format used by every benchmark runner. Demographics, layer identity,
    weights, and all contacts to excluded rows are deliberately discarded.
    """
    if config.get("selection") != "first_n_matrix_rows_induced_subgraph":
        raise ValueError("unsupported GeoPops population selection")
    if config.get("collapse") != "undirected_unweighted_union":
        raise ValueError("unsupported GeoPops collapse rule")
    commit = str(config["source_commit"])
    layers = [str(layer) for layer in config["layers"]]
    hashes = config["source_sha256"]
    source_dir = cache_dir / "geopops" / commit
    raw_base = (
        "https://raw.githubusercontent.com/GeoPopsHub/sc_spartanburg_measles/"
        f"{commit}/data/pop_export"
    )
    sources: dict[str, Path] = {}
    for layer in layers:
        source = source_dir / f"adj_upper_triang_{layer}.mtx"
        _download_verified(
            f"{raw_base}/adj_upper_triang_{layer}.mtx", source, str(hashes[layer])
        )
        sources[layer] = source

    network_dir = cache_dir / "networks"
    stem = f"geopops-collapsed_spartanburg_n{n}_{commit[:12]}"
    edge_path = network_dir / f"{stem}.tsv.gz"
    metadata_path = network_dir / f"{stem}.json"
    expected = {
        "format_version": 1,
        "network_type": "geopops_collapsed",
        "n": n,
        "source_repository": config["source_repository"],
        "source_commit": commit,
        "source_population": config["source_population"],
        "source_layers": layers,
        "source_sha256": {layer: str(hashes[layer]) for layer in layers},
        "selection": config["selection"],
        "collapse": config["collapse"],
        "discarded_features": [
            "contact-layer identity", "demographic attributes", "edge weights",
            "contacts incident to matrix rows outside the selected population",
        ],
    }
    if edge_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(metadata.get(key) == value for key, value in expected.items()):
                if metadata.get("sha256") == sha256_file(edge_path):
                    return edge_path, metadata
        except (OSError, ValueError):
            pass

    edges: set[tuple[int, int]] = set()
    layer_edges: dict[str, int] = {}
    for layer, source in sources.items():
        selected = set(_matrix_market_edges(source, n))
        layer_edges[layer] = len(selected)
        edges.update(selected)

    network_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=network_dir, prefix=f".{edge_path.name}.")
    os.close(fd)
    try:
        with open(temporary, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="ascii", newline="") as stream:
                    stream.write("source\ttarget\n")
                    for source, target in sorted(edges):
                        stream.write(f"{source}\t{target}\n")
        os.replace(temporary, edge_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    edge_count = len(edges)
    metadata = expected | {
        "edges": edge_count,
        "layer_edges_before_union": layer_edges,
        "duplicates_removed_by_union": sum(layer_edges.values()) - edge_count,
        "mean_degree_observed": 2.0 * edge_count / n,
        "density": 2.0 * edge_count / (n * (n - 1)),
        "sha256": sha256_file(edge_path),
    }
    _atomic_json(metadata_path, metadata)
    return edge_path, metadata


def ensure_network_from_config(
    cache_dir: Path, n: int, config: dict, size_index: int = 0
) -> tuple[Path, dict]:
    """Build the network selected by a global or scenario-level table."""
    network_type = config.get("type")
    if network_type == "watts_strogatz":
        return ensure_network(
            cache_dir, n, int(config["mean_degree"]),
            float(config["rewire_probability"]), int(config["seed"]) + size_index,
        )
    if network_type == "geopops_collapsed":
        return ensure_geopops_network(cache_dir, n, config)
    raise ValueError(f"unsupported network type: {network_type!r}")


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
