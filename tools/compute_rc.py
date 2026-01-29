"""
Relative Contrast (RC) evaluation tool with adaptive parameter selection.

Supported inputs:
  - fvecs
  - fbin/u8bin/i8bin
  - ann-benchmarks hdf5
  - npy
  - jsonl/jsonl.gz
  - parquet
  - csv
  - transactions: item IDs per line

Metrics:
  - l2, angular, jaccard, inner-product
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import h5py
import numpy as np

Metric = Literal["l2", "angular", "jaccard", "ip"]


@dataclass
class SampleConfig:
    base_n: int
    query_n: int
    mean_sample_n: int
    seed: int


def _auto_params(base_size: int, query_size: int) -> Tuple[int, int]:
    """
    Adaptive parameter rules (RC-only):
      base_n  = min(base_size, 100_000)
      query_n = min(query_size, 1_000)
    mean_sample_n is set to min(base_n, 20_000)
    """
    base_n = min(base_size, 100_000)
    query_n = min(query_size, 1_000)
    return base_n, query_n


def _is_memmap_view(x: np.ndarray) -> bool:
    return isinstance(x, np.memmap) or isinstance(getattr(x, "base", None), np.memmap)


def _sample_indices(total: int, n_rows: int, seed: int, *, contiguous_if_large: bool = False) -> np.ndarray:
    if n_rows >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if contiguous_if_large:
        start = int(rng.integers(0, total - n_rows))
        return np.arange(start, start + n_rows, dtype=np.int64)
    return rng.choice(total, size=n_rows, replace=False).astype(np.int64, copy=False)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _load_hdf5(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        if "train" not in f or "test" not in f:
            keys = list(f.keys())
            raise KeyError(f"HDF5 missing 'train'/'test'. Found keys: {keys}")
        base = f["train"][:]
        queries = f["test"][:]
    return base, queries


def _memmap_fvecs(path: str) -> Tuple[np.memmap, int, int]:
    with open(path, "rb") as f:
        head = f.read(4)
        if len(head) != 4:
            raise ValueError(f"Empty fvecs file: {path}")
        d = int(np.frombuffer(head, dtype=np.int32, count=1)[0])
        if d <= 0:
            raise ValueError(f"Invalid dimension {d} in {path}")
    size = os.path.getsize(path)
    stride_bytes = 4 * (1 + d)
    if size % stride_bytes != 0:
        raise ValueError(f"Corrupt fvecs file (size not divisible by {stride_bytes} bytes): {path}")
    n = size // stride_bytes
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n, 1 + d))
    return mm, int(n), int(d)


def _load_headered_bin(path: str, dtype: np.dtype) -> np.memmap:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32, count=2)
        if header.shape[0] != 2:
            raise ValueError(f"Invalid header in {path}")
        n, d = int(header[0]), int(header[1])
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def _detect_input_kind(path: str) -> str:
    if os.path.isdir(path):
        raise ValueError(f"Unsupported directory input (please pass a file path): {path}")
    lower = path.lower()
    if lower.endswith((".hdf5", ".h5")):
        return "hdf5"
    if lower.endswith(".fvecs"):
        return "fvecs"
    if lower.endswith(".fbin"):
        return "fbin"
    if lower.endswith(".u8bin"):
        return "u8bin"
    if lower.endswith(".i8bin"):
        return "i8bin"
    if lower.endswith(".npy"):
        return "npy"
    if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
        return "jsonl"
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith((".dat", ".txt")):
        return "transactions"
    raise ValueError(f"Cannot detect input type from path: {path}")


def _is_vector_kind(kind: str) -> bool:
    return kind in {"hdf5", "fvecs", "fbin", "u8bin", "i8bin", "npy", "jsonl", "parquet", "csv"}


def _load_vector_kind(kind: str, path: str, emb_field: str, total_needed: Optional[int], seed: int) -> Tuple[np.ndarray, int]:
    if kind == "fvecs":
        mm, _, d = _memmap_fvecs(path)
        return mm[:, 1:], d
    if kind in {"fbin", "u8bin", "i8bin"}:
        dtype = np.float32 if kind == "fbin" else (np.uint8 if kind == "u8bin" else np.int8)
        mm = _load_headered_bin(path, dtype=np.dtype(dtype))
        return mm, int(mm.shape[1])
    if kind == "npy":
        return _load_npy(path)
    if kind == "jsonl":
        return _load_jsonl_embeddings(path, emb_field, total_needed, seed)
    if kind == "parquet":
        return _load_parquet_embeddings(path, emb_field, total_needed, seed)
    if kind == "csv":
        return _load_csv_embeddings(path, total_needed, seed)
    raise SystemExit(f"Unsupported vector input type: {kind}")


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _load_npy(path: str) -> Tuple[np.ndarray, int]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    return arr, int(arr.shape[1])


def _load_jsonl_embeddings(path: str, emb_field: str, total_needed: Optional[int], seed: int) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    sample: list[np.ndarray] = []
    seen = 0
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            emb = obj.get(emb_field)
            if emb is None:
                continue
            vec = np.asarray(emb, dtype=np.float32)
            if vec.ndim != 1:
                vec = vec.reshape(-1)
            seen += 1
            if total_needed is None:
                sample.append(vec)
            elif len(sample) < total_needed:
                sample.append(vec)
            else:
                j = int(rng.integers(0, seen))
                if j < total_needed:
                    sample[j] = vec
    if not sample:
        raise ValueError(f"No embeddings found in {path} (field '{emb_field}')")
    mat = np.vstack(sample).astype(np.float32, copy=False)
    return mat, int(mat.shape[1])


def _load_parquet_embeddings(path: str, emb_field: str, total_needed: Optional[int], seed: int) -> Tuple[np.ndarray, int]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("pyarrow is required for parquet inputs. Install with: pip install pyarrow") from exc

    rng = np.random.default_rng(seed)
    sample: list[np.ndarray] = []
    seen = 0
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[emb_field])
        col = table.column(0).to_pylist()
        for emb in col:
            if emb is None:
                continue
            vec = np.asarray(emb, dtype=np.float32)
            if vec.ndim != 1:
                vec = vec.reshape(-1)
            seen += 1
            if total_needed is None:
                sample.append(vec)
            elif len(sample) < total_needed:
                sample.append(vec)
            else:
                j = int(rng.integers(0, seen))
                if j < total_needed:
                    sample[j] = vec
    if not sample:
        raise ValueError(f"No embeddings found in {path} (field '{emb_field}')")
    mat = np.vstack(sample).astype(np.float32, copy=False)
    return mat, int(mat.shape[1])


def _load_csv_embeddings(path: str, total_needed: Optional[int], seed: int) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    sample: list[np.ndarray] = []
    seen = 0
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            _, emb_str = line.split("\t", 1)
            vec = np.fromstring(emb_str, sep=",", dtype=np.float32)
            if vec.size == 0:
                continue
            seen += 1
            if total_needed is None:
                sample.append(vec)
            elif len(sample) < total_needed:
                sample.append(vec)
            else:
                j = int(rng.integers(0, seen))
                if j < total_needed:
                    sample[j] = vec
    if not sample:
        raise ValueError(f"No embeddings found in {path}")
    mat = np.vstack(sample).astype(np.float32, copy=False)
    return mat, int(mat.shape[1])

def _reservoir_sample_transactions(path: str, total_needed: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample: list[np.ndarray] = []
    seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                items = np.fromiter((int(x) for x in parts), dtype=np.int32, count=len(parts))
            except Exception:
                continue
            if items.size == 0:
                continue
            items = np.unique(items)
            seen += 1
            if len(sample) < total_needed:
                sample.append(items)
            else:
                j = int(rng.integers(0, seen))
                if j < total_needed:
                    sample[j] = items
    if len(sample) < total_needed:
        raise ValueError(f"Transaction file too small after parsing: got {len(sample)} rows, need {total_needed}")
    arr = np.empty((total_needed,), dtype=object)
    for i, v in enumerate(sample):
        arr[i] = v
    return arr


def _jaccard_distance_sorted_int(a: np.ndarray, b: np.ndarray) -> float:
    i = j = 0
    inter = 0
    na = a.shape[0]
    nb = b.shape[0]
    while i < na and j < nb:
        va = int(a[i])
        vb = int(b[j])
        if va == vb:
            inter += 1
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    union = na + nb - inter
    if union <= 0:
        return 0.0
    return 1.0 - (inter / union)


def _compute_jaccard_dmin_dmean(queries: np.ndarray, base_all: np.ndarray, base_mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    qn = queries.shape[0]
    dmin = np.empty((qn,), dtype=np.float64)
    dmean = np.empty((qn,), dtype=np.float64)
    for qi in range(qn):
        q = queries[qi]
        best = 1.0
        for b in base_all:
            d = _jaccard_distance_sorted_int(q, b)
            if d < best:
                best = d
                if best <= 0.0:
                    break
        dmin[qi] = best
        acc = 0.0
        for b in base_mean:
            acc += _jaccard_distance_sorted_int(q, b)
        dmean[qi] = acc / float(len(base_mean))
    return dmin, dmean


def _min_distances_chunked(
    queries: np.ndarray,
    base: np.ndarray,
    metric: Metric,
    *,
    chunk_q: int = 128,
    chunk_b: int = 50_000,
    query_ids: Optional[np.ndarray] = None,
    base_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    qn = queries.shape[0]
    dmin = np.empty((qn,), dtype=np.float64)

    for start in range(0, qn, chunk_q):
        end = min(start + chunk_q, qn)
        q = queries[start:end].astype(np.float32, copy=False)
        best = np.full((end - start,), np.inf, dtype=np.float64)

        for bs in range(0, base.shape[0], chunk_b):
            be = min(bs + chunk_b, base.shape[0])
            b = base[bs:be]

            if metric == "l2":
                q2 = np.sum(q * q, axis=1, keepdims=True)
                b2 = np.sum(b * b, axis=1, keepdims=True).T
                d2 = q2 + b2 - 2.0 * (q @ b.T)
                np.maximum(d2, 0.0, out=d2)
                d = np.sqrt(d2, dtype=np.float64)
            elif metric == "angular":
                qnrm = _l2_normalize(q)
                bnrm = _l2_normalize(b)
                sim = qnrm @ bnrm.T
                d = (1.0 - sim).astype(np.float64, copy=False)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if query_ids is not None and base_ids is not None:
                block_ids = base_ids[bs:be]
                for qi in range(end - start):
                    qid = int(query_ids[start + qi])
                    pos = np.searchsorted(block_ids, qid)
                    if 0 <= pos < (be - bs) and int(block_ids[pos]) == qid:
                        d[qi, pos] = np.inf

            best = np.minimum(best, np.min(d, axis=1))

        dmin[start:end] = best

    return dmin


def _mean_distances_chunked(
    queries: np.ndarray,
    base: np.ndarray,
    metric: Metric,
    *,
    chunk_q: int = 128,
    chunk_b: int = 50_000,
) -> np.ndarray:
    qn = queries.shape[0]
    dmean = np.empty((qn,), dtype=np.float64)

    for start in range(0, qn, chunk_q):
        end = min(start + chunk_q, qn)
        q = queries[start:end].astype(np.float32, copy=False)
        acc = np.zeros((end - start,), dtype=np.float64)
        total = 0

        for bs in range(0, base.shape[0], chunk_b):
            be = min(bs + chunk_b, base.shape[0])
            b = base[bs:be]
            total += (be - bs)

            if metric == "l2":
                q2 = np.sum(q * q, axis=1, keepdims=True)
                b2 = np.sum(b * b, axis=1, keepdims=True).T
                d2 = q2 + b2 - 2.0 * (q @ b.T)
                np.maximum(d2, 0.0, out=d2)
                d = np.sqrt(d2, dtype=np.float64)
            elif metric == "angular":
                qnrm = _l2_normalize(q)
                bnrm = _l2_normalize(b)
                sim = qnrm @ bnrm.T
                d = (1.0 - sim).astype(np.float64, copy=False)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            acc += np.sum(d, axis=1)

        dmean[start:end] = acc / float(total)

    return dmean


def _estimate_rc(
    base: np.ndarray,
    queries: np.ndarray,
    metric: Metric,
    cfg: SampleConfig,
) -> dict:
    base_total = int(base.shape[0])
    query_total = int(queries.shape[0])

    base_contig = _is_memmap_view(base) and base_total > 1_000_000
    query_contig = _is_memmap_view(queries) and query_total > 1_000_000

    base_ids = _sample_indices(base_total, min(cfg.base_n, base_total), cfg.seed, contiguous_if_large=base_contig)
    query_ids = _sample_indices(query_total, min(cfg.query_n, query_total), cfg.seed + 1, contiguous_if_large=query_contig)
    base_ids.sort()
    query_ids.sort()

    base_s = base[base_ids]
    q_s = queries[query_ids]

    if base_s.shape[0] > cfg.mean_sample_n:
        mean_ids = _sample_indices(int(base_s.shape[0]), cfg.mean_sample_n, cfg.seed + 2, contiguous_if_large=False)
        base_mean = base_s[mean_ids]
    else:
        base_mean = base_s

    exclude_self = False
    try:
        exclude_self = np.shares_memory(base, queries)
    except Exception:
        exclude_self = base is queries

    if metric == "jaccard":
        dmin, dmean = _compute_jaccard_dmin_dmean(q_s, base_s, base_mean)
        rc = float(np.mean(dmean) / np.mean(dmin))
        note = None
    elif metric == "ip":
        base_s = np.asarray(base_s, dtype=np.float32)
        q_s = np.asarray(q_s, dtype=np.float32)
        smax = np.empty((q_s.shape[0],), dtype=np.float64)
        smean = np.empty((q_s.shape[0],), dtype=np.float64)
        for start in range(0, q_s.shape[0], 128):
            end = min(start + 128, q_s.shape[0])
            q = q_s[start:end]
            sim_mean = (q @ base_mean.T).astype(np.float64)
            smean[start:end] = np.mean(sim_mean, axis=1)
            sim_all = (q @ base_s.T).astype(np.float64)
            if exclude_self:
                for qi in range(end - start):
                    qid = int(query_ids[start + qi])
                    pos = np.searchsorted(base_ids, qid)
                    if 0 <= pos < base_s.shape[0] and int(base_ids[pos]) == qid:
                        sim_all[qi, pos] = -np.inf
            smax[start:end] = np.max(sim_all, axis=1)
        rc = float(np.mean(smax) / np.mean(smean))
        note = "metric=ip: rc=E[S_max]/E[S_mean]"
    else:
        base_s = np.asarray(base_s, dtype=np.float32)
        q_s = np.asarray(q_s, dtype=np.float32)
        dmin = _min_distances_chunked(
            q_s,
            base_s,
            metric,
            query_ids=(query_ids if exclude_self else None),
            base_ids=(base_ids if exclude_self else None),
        )
        dmean = _mean_distances_chunked(q_s, base_mean.astype(np.float32), metric)
        rc = float(np.mean(dmean) / np.mean(dmin))
        note = None

    return {
        "metric": metric,
        "rc": rc,
        "base_used": int(base_s.shape[0]),
        "queries_used": int(q_s.shape[0]),
        "mean_base_used": int(base_mean.shape[0]),
        "seed": int(cfg.seed),
        "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=["l2", "angular", "jaccard", "ip"])
    ap.add_argument("--auto", action="store_true", help="Use adaptive base/query rules.")
    ap.add_argument("--base-n", type=int, default=20_000)
    ap.add_argument("--query-n", type=int, default=200)
    ap.add_argument("--mean-sample-n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--base", type=str, help="Auto-detect input type from path.")
    ap.add_argument("--query", type=str, help="Auto-detect query input type from path.")
    ap.add_argument("--hdf5", type=str)
    ap.add_argument("--fvecs-base", type=str)
    ap.add_argument("--fvecs-query", type=str)
    ap.add_argument("--base-bin", type=str)
    ap.add_argument("--query-bin", type=str)
    ap.add_argument("--bin-dtype", type=str, choices=["f32", "i8", "u8"], default="f32")
    ap.add_argument("--npy-base", type=str)
    ap.add_argument("--npy-query", type=str)
    ap.add_argument("--jsonl-base", type=str)
    ap.add_argument("--jsonl-query", type=str)
    ap.add_argument("--jsonl-emb-field", type=str, default="emb")
    ap.add_argument("--parquet-base", type=str)
    ap.add_argument("--parquet-query", type=str)
    ap.add_argument("--parquet-emb-field", type=str, default="emb")
    ap.add_argument("--transactions", type=str)
    ap.add_argument("--ml-10m", type=str)
    ap.add_argument("--ml-10m-cache", action="store_true")
    args = ap.parse_args()

    metric: Metric = args.metric

    base = None
    queries = None

    if args.base or args.query:
        base_path = args.base
        query_path = args.query
        if not base_path:
            raise SystemExit("--base is required when using auto-detect input.")

        kind = _detect_input_kind(base_path)
        if kind == "hdf5":
            if query_path:
                raise SystemExit("HDF5 already contains queries; do not set --query.")
            base, queries = _load_hdf5(base_path)
        elif _is_vector_kind(kind):
            base_emb_field = args.parquet_emb_field if kind == "parquet" else args.jsonl_emb_field
            base, base_d = _load_vector_kind(kind, base_path, base_emb_field, args.base_n, args.seed)
            if query_path:
                q_kind = _detect_input_kind(query_path)
                if not _is_vector_kind(q_kind):
                    raise SystemExit(f"Query input type '{q_kind}' is not vector-compatible.")
                queries, query_d = _load_vector_kind(q_kind, query_path, args.jsonl_emb_field, args.query_n, args.seed + 1)
                if base_d != query_d:
                    raise SystemExit(f"Dimension mismatch: base d={base_d}, query d={query_d}")
            else:
                queries = base
        elif kind == "transactions":
            if metric != "jaccard":
                raise SystemExit("Transaction input requires --metric jaccard.")
            if query_path:
                if _detect_input_kind(query_path) != "transactions":
                    raise SystemExit("Transaction input requires transaction-format query.")
                base = _reservoir_sample_transactions(base_path, total_needed=args.base_n, seed=args.seed)
                queries = _reservoir_sample_transactions(query_path, total_needed=args.query_n, seed=args.seed + 1)
            else:
                total = args.base_n + args.query_n
                arr = _reservoir_sample_transactions(base_path, total_needed=total, seed=args.seed)
                base = arr[: args.base_n]
                queries = arr[args.base_n :]
        elif kind == "ml-10m":
            if metric != "jaccard":
                raise SystemExit("MovieLens-10M input requires --metric jaccard.")
            if query_path:
                raise SystemExit("MovieLens-10M expects a single directory input; do not set --query.")
            ratings = os.path.join(base_path, "ratings.dat")
            cache_path = os.path.join(base_path, "user_movies_cache.npz")
            if args.ml_10m_cache and os.path.exists(cache_path):
                d = np.load(cache_path, allow_pickle=True)
                arr = d["user_movie_sets"]
            else:
                user_movies: dict[int, set[int]] = {}
                with open(ratings, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split("::")
                        if len(parts) < 2:
                            continue
                        try:
                            uid = int(parts[0])
                            mid = int(parts[1])
                        except Exception:
                            continue
                        s = user_movies.get(uid)
                        if s is None:
                            s = set()
                            user_movies[uid] = s
                        s.add(mid)
                users = sorted(user_movies.keys())
                arr = np.empty((len(users),), dtype=object)
                for idx, uid in enumerate(users):
                    arr[idx] = np.array(sorted(user_movies[uid]), dtype=np.int32)
                if args.ml_10m_cache:
                    np.savez_compressed(cache_path, user_movie_sets=arr)
            base = arr
            queries = arr
        else:
            raise SystemExit(f"Unsupported input type: {kind}")
    elif args.hdf5:
        base, queries = _load_hdf5(args.hdf5)
    elif args.fvecs_base:
        if not args.fvecs_query:
            raise SystemExit("fvecs requires --fvecs-query")
        base_mm, _, base_d = _memmap_fvecs(args.fvecs_base)
        query_mm, _, query_d = _memmap_fvecs(args.fvecs_query)
        if base_d != query_d:
            raise SystemExit(f"Dimension mismatch: base d={base_d}, query d={query_d}")
        base = base_mm[:, 1:]
        queries = query_mm[:, 1:]
    elif args.npy_base or args.jsonl_base or args.parquet_base:
        if args.npy_base:
            base, base_d = _load_npy(args.npy_base)
        elif args.jsonl_base:
            base, base_d = _load_jsonl_embeddings(args.jsonl_base, args.jsonl_emb_field, args.base_n, args.seed)
        else:
            base, base_d = _load_parquet_embeddings(args.parquet_base, args.parquet_emb_field, args.base_n, args.seed)

        if args.npy_query:
            queries, query_d = _load_npy(args.npy_query)
        elif args.jsonl_query:
            queries, query_d = _load_jsonl_embeddings(args.jsonl_query, args.jsonl_emb_field, None, args.seed + 1)
        elif args.parquet_query:
            queries, query_d = _load_parquet_embeddings(args.parquet_query, args.parquet_emb_field, None, args.seed + 1)
        else:
            queries, query_d = base, base_d

        if base_d != query_d:
            raise SystemExit(f"Dimension mismatch: base d={base_d}, query d={query_d}")
    elif args.base_bin:
        dtype = np.float32 if args.bin_dtype == "f32" else (np.int8 if args.bin_dtype == "i8" else np.uint8)
        base_mm = _load_headered_bin(args.base_bin, dtype=np.dtype(dtype))
        if args.query_bin:
            query_mm = _load_headered_bin(args.query_bin, dtype=np.dtype(dtype))
            base = base_mm
            queries = query_mm
        else:
            base = base_mm
            queries = base_mm
    elif args.transactions:
        total = args.base_n + args.query_n
        arr = _reservoir_sample_transactions(args.transactions, total_needed=total, seed=args.seed)
        base = arr[: args.base_n]
        queries = arr[args.base_n :]
    elif args.ml_10m:
        ratings = os.path.join(args.ml_10m, "ratings.dat")
        if not os.path.exists(ratings):
            raise SystemExit(f"ratings.dat not found under {args.ml_10m}")
        cache_path = os.path.join(args.ml_10m, "user_movies_cache.npz")
        if args.ml_10m_cache and os.path.exists(cache_path):
            d = np.load(cache_path, allow_pickle=True)
            arr = d["user_movie_sets"]
        else:
            user_movies: dict[int, set[int]] = {}
            with open(ratings, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("::")
                    if len(parts) < 2:
                        continue
                    try:
                        uid = int(parts[0])
                        mid = int(parts[1])
                    except Exception:
                        continue
                    s = user_movies.get(uid)
                    if s is None:
                        s = set()
                        user_movies[uid] = s
                    s.add(mid)
            users = sorted(user_movies.keys())
            arr = np.empty((len(users),), dtype=object)
            for idx, uid in enumerate(users):
                arr[idx] = np.array(sorted(user_movies[uid]), dtype=np.int32)
            if args.ml_10m_cache:
                np.savez_compressed(cache_path, user_movie_sets=arr)
        base = arr
        queries = arr
    else:
        raise SystemExit("No input specified.")

    base_size = int(base.shape[0])
    query_size = int(queries.shape[0])

    if args.auto:
        base_n, query_n = _auto_params(base_size, query_size)
        mean_sample_n = min(base_n, 20_000)
    else:
        base_n, query_n = args.base_n, args.query_n
        mean_sample_n = args.mean_sample_n

    cfg = SampleConfig(
        base_n=base_n,
        query_n=query_n,
        mean_sample_n=mean_sample_n,
        seed=args.seed,
    )

    res = _estimate_rc(base, queries, metric, cfg)
    result = {
        "rc": res["rc"],
        "base_used": res["base_used"],
        "queries_used": res["queries_used"],
        "mean_base_used": res["mean_base_used"],
    }
    if res.get("note") is not None:
        result["note"] = res.get("note")
    output = {
        "params": {
            "metric": metric,
            "base_n": cfg.base_n,
            "query_n": cfg.query_n,
            "mean_sample_n": cfg.mean_sample_n,
            "seed": cfg.seed,
            "auto": bool(args.auto),
        },
        "result": result,
    }
    print(output)


if __name__ == "__main__":
    main()


