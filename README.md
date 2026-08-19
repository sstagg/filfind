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
- `extract_filament_endpoints.py`: recover filament endpoints from a cleaned particle STAR that still has filament IDs.
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
  --overlay-max-dim 1024 \
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
  --overlay-max-dim 1024 \
  --output-dir /path/to/out
```

## Recover endpoints from cleaned particles

After 2D/3D cleanup, you may have good particle picks from each filament but no original start/end picks. If the particle STAR still contains `_rlnMicrographName` and a filament identity column such as `_rlnHelicalTubeID`, this command groups picks by micrograph and filament, finds the two picks farthest apart in each filament, and writes recovered endpoint picks back into per-micrograph coordinate STAR files:

```bash
python extract_filament_endpoints.py \
  --input /path/to/clean_particles.star \
  --out-dir /path/to/endpoint_coords \
  --output /path/to/endpoint_coords/filfind_coordinate_files.star
```

The `--output` file is a RELION-style coordinate-files STAR with `_rlnMicrographName` and `_rlnMicrographCoordinates`; each referenced coordinate STAR contains the recovered start/end coordinates for one micrograph.

By default, the script auto-detects common RELION-style filament columns:

- `_rlnHelicalTubeID`
- `_rlnFilamentID`
- `_rlnFilamentNumber`
- `_rlnTubeID`
- `_rlnHelicalTubeName`

If your STAR uses a different column, pass it explicitly:

```bash
python extract_filament_endpoints.py \
  --input /path/to/clean_particles.star \
  --filament-column _rlnMyFilamentColumn \
  --out-dir /path/to/endpoint_coords
```

If you want an extra combined endpoint STAR for inspection, add `--single-star /path/to/all_endpoints.star`.

Singleton classified particles can optionally be mapped back to the original filfind endpoints. Groups with 2+ classified particles still use the farthest classified pair; groups with exactly 1 classified particle keep the nearest original endpoint pair when the singleton lies close to that original finite segment:

```bash
python extract_filament_endpoints.py \
  --input /path/to/clean_particles.star \
  --out-dir /path/to/endpoint_coords \
  --singleton-parent-endpoint-dir /path/to/original/filfind/out \
  --singleton-max-distance-px 5
```

To render binned overlay PNGs next to the output STAR files, add `--mrc-dir /path/to/MotionCorr/jobXXX/frames`. By default this renders every matching image; add `--max-images N` to limit the run.

Notes:
- `--output-dir` and `--out-dir` are equivalent.
- Overlay PNGs are downscaled to max dimension `1024` by default (`--overlay-max-dim`).
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
