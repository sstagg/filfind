# filfind

Filament tracing utilities for cryo-EM micrographs using Topaz autopicks.

## What this repo does

- Reads Topaz `*_autopick.star` files and corresponding `.mrc` images.
- Filters picks by FOM.
- Builds candidate neighbor graph from pairwise distances.
- Greedily traces filaments under line-fit constraints.
- Writes:
  - endpoint STAR files for downstream processing,
  - optional overlay PNGs,
  - optional NPZ summary.

## Environment

```bash
conda create -n findfil python=3.11 -y
conda activate findfil
pip install numpy matplotlib mrcfile starfile
```

## Main scripts

- `trace_filaments.py`: run tracing on one STAR/MRC pair.
- `trace_filaments_relion_job.py`: batch over a RELION-style job tree.
- `filfind_trace_lib.py`: shared tracing logic.

## Single-file run

```bash
python trace_filaments.py \
  --autopick /path/to/file_autopick.star \
  --mrc /path/to/file.mrc \
  --fom-min 0 \
  --candidate-k-std 1.5 \
  --max-neighbors 2 \
  --max-line-rms 14 \
  --out-dir /path/to/out
```

## RELION batch run

```bash
python trace_filaments_relion_job.py \
  --job-dir /path/to/AutoPick/jobXXX/frames \
  --mrc-root /path/to/MotionCorr/jobYYY/frames \
  --fom-min -0.5 \
  --candidate-k-std 1.5 \
  --max-neighbors 2 \
  --max-line-rms 14 \
  --output-dir /path/to/out
```

Notes:
- `--output-dir` and `--out-dir` are equivalent.
- For multiline shell commands, each continued line must end with `\`.
- In batch mode, outputs are forced under `--output-dir` when provided.

## Batch command log

Batch runs write a human-readable command log with status for each file.

Default log location:
- if `--output-dir` is set: `<output-dir>/filfind_batch_commands.log`
- otherwise: `<job-dir>/filfind_batch_commands.log`

You can override with:

```bash
--command-log /path/to/custom.log
```

## Typical outputs

For each processed micrograph stem `X`:

- `X_filfind_endpoints.star`
- `X_filfind_overlay.png` (unless `--no-save-overlay`)
- `X_filfind.npz` (only if `--save-npz`)
