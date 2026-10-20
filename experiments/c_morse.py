"""Experiment C: apparent pairs, Morse parity, and Adcock-Carlsson coordinates.

NumPy builds the unreduced boundary. A CUDA scan (Triton if it compiles, otherwise
torch) recomputes the column lows. GDV uses one local diagram per labeled point.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn

from laplacian.device import enforce_guard, pick_5070
from laplacian.mcp_tools import gpu_telemetry_and_guard, verification_gate
from laplacian.runtime import apply_env, write_metrics

MONOMIALS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (2, 2),
    (3, 1),
    (2, 3),
    (3, 2),
)


def _complex(points):
    count = len(points)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    simplices = [(0, (index,), 0.0) for index in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            simplices.append((1, (left, right), float(dist[left, right])))
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                filt = float(max(dist[i, j], dist[i, k], dist[j, k]))
                simplices.append((2, (i, j, k), filt))
    simplices.sort(key=lambda item: (item[2], item[0], item[1]))
    index = {item[1]: position for position, item in enumerate(simplices)}
    return simplices, index


def _facets(vertices):
    if len(vertices) == 1:
        return []
    found = []
    for drop in range(len(vertices)):
        found.append(vertices[:drop] + vertices[drop + 1 :])
    return found


def _boundary_bits(simplices, index):
    columns = []
    for _dim, vertices, _filt in simplices:
        column = 0
        for facet in _facets(vertices):
            column |= 1 << index[facet]
        columns.append(column)
    return columns


def _apparent_pairs(simplices, index):
    cofacets = {position: [] for position in range(len(simplices))}
    for position, (dim, vertices, _filt) in enumerate(simplices):
        if dim == 0:
            continue
        for facet in _facets(vertices):
            cofacets[index[facet]].append(position)
    pairs = []
    for position, (dim, vertices, _filt) in enumerate(simplices):
        if dim == 0:
            continue
        facet_ids = [index[facet] for facet in _facets(vertices)]
        youngest = max(facet_ids, key=lambda item: (simplices[item][2], item))
        cofacet_ids = cofacets[youngest]
        oldest = min(cofacet_ids, key=lambda item: (simplices[item][2], item))
        if oldest == position:
            pairs.append((youngest, position))
    return pairs


def _low(column):
    if column == 0:
        return -1
    return column.bit_length() - 1


def _reduce(columns):
    pivots = {}
    finite = []
    for column_index, original in enumerate(columns):
        column = original
        while column and _low(column) in pivots:
            column ^= pivots[_low(column)]
        pivot = _low(column)
        if pivot >= 0:
            pivots[pivot] = column
            finite.append((pivot, column_index))
    return finite


def _diagram(simplices, pairs):
    bars = []
    for birth, death in pairs:
        start = simplices[birth][2]
        end = simplices[death][2]
        if end > start:
            bars.append((start, end, simplices[birth][0]))
    return bars


def adcock_carlsson(bars):
    values = []
    for birth_power, pers_power in MONOMIALS:
        total = 0.0
        for birth, death, _dim in bars:
            persistence = death - birth
            total += (birth ** birth_power) * (persistence ** pers_power)
        values.append(total)
    return values


def _vpath_parity(simplices, pairs, depth_cap=6):
    """Capped parity of alternating Morse paths. The walk is an explicit stack."""
    lookup = {item[1]: position for position, item in enumerate(simplices)}
    matched_up = {facet: cofacet for facet, cofacet in pairs}
    matched_down = {cofacet: facet for facet, cofacet in pairs}
    critical = {
        index
        for index in range(len(simplices))
        if index not in matched_up and index not in matched_down
    }
    parity = 0
    seen_paths = 0
    for start in critical:
        stack = [(start, 0)]
        visited = {start}
        while stack and seen_paths < 100000:
            simplex, depth = stack.pop()
            if depth >= depth_cap:
                continue
            for facet_vertices in _facets(simplices[simplex][1]):
                facet = lookup.get(facet_vertices)
                if facet is None or facet == matched_down.get(simplex):
                    continue
                nxt = matched_up.get(facet)
                if nxt is None:
                    continue
                if nxt in critical and nxt != start:
                    parity ^= 1
                    seen_paths += 1
                elif nxt not in visited and depth + 1 < depth_cap:
                    visited.add(nxt)
                    stack.append((nxt, depth + 1))
    return {
        "apparent_pairs": len(pairs),
        "critical_cells": len(critical),
        "vpath_parity": int(parity),
        "vpath_count_capped": int(seen_paths),
        "depth_cap": depth_cap,
    }


def _knn_points(points, labels, neighbors):
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    features = []
    for row in range(len(points)):
        chosen = np.argsort(dist[row])[:neighbors]
        local, index = _complex(points[chosen])
        columns = _boundary_bits(local, index)
        bars = _diagram(local, _reduce(columns))
        features.append(adcock_carlsson(bars))
    return np.asarray(features, dtype=np.float64), labels.copy()


def _gdv(features, labels):
    array = np.asarray(features, dtype=np.float64)
    width = max(array.shape[1], 1)
    intra = []
    inter = []
    for left in range(len(array)):
        for right in range(left + 1, len(array)):
            distance = float(np.linalg.norm(array[left] - array[right]))
            if labels[left] == labels[right]:
                intra.append(distance)
            else:
                inter.append(distance)
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return (intra_mean - inter_mean) / np.sqrt(width)


def _mlp_features(points, labels, device):
    module = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    module.to(device)
    data = torch.tensor(points, dtype=torch.float32, device=device)
    target = torch.tensor(labels, dtype=torch.long, device=device)
    opt = torch.optim.SGD(module.parameters(), lr=0.05)
    module.train()
    for _step in range(200):
        opt.zero_grad(set_to_none=True)
        logits = module(data)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        opt.step()
    module.eval()
    with torch.no_grad():
        hidden = torch.relu(module[2](torch.relu(module[0](data))))
    return hidden.detach().cpu().numpy(), float(loss.detach().cpu())


def _scan_lows(matrix, device):
    started = time.perf_counter()
    weighted = matrix.astype(np.int32) * np.arange(matrix.shape[0], dtype=np.int32)[:, None]
    numpy_lows = weighted.max(axis=0)
    numpy_lows[matrix.sum(axis=0) == 0] = -1
    numpy_seconds = time.perf_counter() - started
    backend = "torch"
    triton_error = ""
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _low_kernel(src, out, n, BLOCK: tl.constexpr):
            col = tl.program_id(0)
            lowest = -1
            for start in range(0, n, BLOCK):
                rows = start + tl.arange(0, BLOCK)
                mask = rows < n
                values = tl.load(src + rows * n + col, mask=mask, other=0)
                hits = tl.where(values > 0, rows, -1)
                lowest = tl.maximum(lowest, tl.max(hits, axis=0))
            tl.store(out + col, lowest)

        packed = torch.from_numpy(matrix.astype(np.int32)).to(device)
        result = torch.empty(matrix.shape[0], dtype=torch.int32, device=device)
        _low_kernel[(matrix.shape[0],)](packed, result, matrix.shape[0], BLOCK=128)
        torch.cuda.synchronize()
        backend = "triton"
        cuda_lows = result.detach().cpu().numpy()
    except Exception as exc:
        triton_error = str(exc)[:400]
        gpu = torch.from_numpy(matrix.astype(np.int32)).to(device)
        index = torch.arange(matrix.shape[0], device=device, dtype=torch.int32)
        weighted_gpu = gpu * index[:, None]
        cuda_lows = weighted_gpu.amax(dim=0)
        empty = gpu.sum(dim=0) == 0
        cuda_lows = torch.where(empty, torch.full_like(cuda_lows, -1), cuda_lows).detach().cpu().numpy()
    return {
        "backend": backend,
        "triton_error": triton_error,
        "numpy_seconds": numpy_seconds,
        "max_abs_low_gap": int(np.max(np.abs(numpy_lows.astype(np.int64) - cuda_lows.astype(np.int64)))),
    }


def main():
    apply_env()
    device, _name = pick_5070()
    telemetry = gpu_telemetry_and_guard()
    rng = np.random.default_rng(0)
    class0 = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(20, 2))
    class1 = rng.normal(loc=(2.5, 0.0), scale=1.0, size=(20, 2))
    points = np.vstack([class0, class1])
    labels = np.array([0] * 20 + [1] * 20)
    started = time.perf_counter()
    simplices, index = _complex(points)
    columns = _boundary_bits(simplices, index)
    pairs = _apparent_pairs(simplices, index)
    finite = _reduce(columns)
    bars = _diagram(simplices, finite)
    morse = _vpath_parity(simplices, pairs)
    numpy_build_seconds = time.perf_counter() - started
    count = len(simplices)
    matrix = np.zeros((count, count), dtype=np.uint8)
    for column, (dim, vertices, _filt) in enumerate(simplices):
        for facet in _facets(vertices):
            matrix[index[facet], column] = 1
    scan = _scan_lows(matrix, device)
    features, feature_labels = _knn_points(points, labels, neighbors=8)
    morse_gdv = float(_gdv(features, feature_labels))
    mlp_hidden, mlp_loss = _mlp_features(points, labels, device)
    mlp_gdv = float(_gdv(mlp_hidden, labels))
    gate = verification_gate("gdv", {"morse": morse_gdv, "mlp": mlp_gdv})
    enforce_guard()
    payload = {
        "experiment": "C",
        "host_gpu": True,
        "complete": True,
        "device": telemetry,
        "n_points": 40,
        "classes": "two isotropic Gaussians, means (0,0) and (2.5,0)",
        "n_simplices": count,
        "apparent_pairs": morse["apparent_pairs"],
        "critical_cells": morse["critical_cells"],
        "vpath_parity": morse["vpath_parity"],
        "vpath_count_capped": morse["vpath_count_capped"],
        "vpath_depth_cap": morse["depth_cap"],
        "finite_bars": len(bars),
        "global_adcock_carlsson": adcock_carlsson(bars),
        "monomials_birth_persistence": [list(item) for item in MONOMIALS],
        "knn": 8,
        "gdv_definition": "(mean intra-class L2 distance - mean inter-class L2 distance) / sqrt(dimension)",
        "measured_gdv_morse": morse_gdv,
        "measured_gdv_mlp": mlp_gdv,
        "mlp_final_loss": mlp_loss,
        "gdv_gate": gate,
        "scan": scan,
        "numpy_complex_seconds": numpy_build_seconds,
        "seconds": time.perf_counter() - started,
        "peak_vram_mib": enforce_guard(),
    }
    digest = write_metrics(ROOT / "runs" / "guidance_c" / "metrics.json", payload)
    print(f"DONE guidance_c sha256={digest} backend={scan['backend']} gdv_gap={morse_gdv - mlp_gdv}", flush=True)


if __name__ == "__main__":
    main()
