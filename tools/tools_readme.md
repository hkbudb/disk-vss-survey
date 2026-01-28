# RC/LID Tools

This folder contains two evaluation tools:

- `compute_rc.py`: computes Relative Contrast (RC)
- `compute_lid.py`: computes Local Intrinsic Dimensionality (LID)

Both tools support the same input formats and metrics, and both can use adaptive sampling via `--auto`.

## Quick start
```bash
python tools/compute_rc.py --base <base_file.ext> --query <query_file.ext> --metric <metric> --auto
python tools/compute_lid.py --base <base_file.ext> --query <query_file.ext> --metric <metric> --auto
```

## Parameters

### Input (auto-detect)
- `--base <file>` and optional `--query <file>`
- Supported extensions: `.hdf5/.h5`, `.fvecs`, `.fbin`, `.u8bin`, `.i8bin`, `.npy`, `.jsonl[.gz]`, `.parquet`, `.csv`, `.dat/.txt`
- For JSONL/Parquet embeddings, set `--jsonl-emb-field` or `--parquet-emb-field` (default `emb`)

### Metrics
- `--metric <l2|angular|ip|jaccard>`

### Sampling / LID
- `--auto` enables adaptive sampling
  - RC: `base_n`, `query_n`, `mean_sample_n`
  - LID: `base_n`, `query_n`, `k_lid`
- Manual examples:
  - RC: `--base-n 20000 --query-n 200 --mean-sample-n 20000 --seed 42`
  - LID: `--base-n 20000 --query-n 200 --k-lid 50 --seed 42`

## Output

Each tool prints a single JSON-like dict with:

- `params`: input parameters actually used
- `result`: metric result and effective sample sizes

RC `result` includes `rc`, `base_used`, `queries_used`, `mean_base_used`. 
LID `result` includes `lid`, `base_used`, `queries_used`, `k_lid`.

## Notes

- If only a base file is provided, the tool samples queries from the base pool.
- For `metric=ip`:
  - RC uses similarities: `RC_ip = E[S_max] / E[S_mean]`
  - LID uses positive distance gaps: `r_i = s_max - s_{i+1}` 
