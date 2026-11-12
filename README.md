# CS2 win probability

Prepare one Parquet file of labeled game states from a CS2 demo.

```sh
uv run cs-win-prob match.dem states.parquet --ticks-per-round 4 --seed 42
```

Sampling defaults to four distinct random ticks per round, from freeze end
through the tick that decides the round, inclusive. Short or sparse rounds
contribute as many states as are available up to that limit. Rounds without
an observed playable interval or a supported outcome are skipped.

```python
from awpy import Demo
from cs_win_prob.data_preparation import prepare_demo

states = prepare_demo(Demo("match.dem"), ticks_per_round=4, seed=42)
states.write_parquet("states.parquet")
```

`map` uses `other` for maps outside the training set. Pass `training_maps` to
`prepare_demo` to use a different set. `phase` is `bomb_not_planted` or
`bomb_planted`, and `time_since_phase_start` is in seconds from freeze end or
the completed plant, respectively.

Each row includes `t_` and `ct_` counts for dead players, health and armor
bands, helmets, fire grenades (Molotovs and incendiaries), smokes, HE grenades,
assault rifles, SMGs, and AWPs, plus each side's total equipment value and
`ct_defuse_kits`. The health and armor bands are 0–24, 25–49, 50–74, and 75–100.
Only living players contribute to the health and armor bands, equipment counts,
and equipment value. The table also includes total live HP and carried flashbangs
per side, the planted bombsite, and whether a live CT is defusing.

`ct_distance_to_bomb` and `t_distance_to_bomb` are the shortest 3D straight-line
distances from a living player on each side, in Hammer units. They use the
planter's position at plant completion as the C4 location estimate. Distances
are null before the plant or when the position or a living player is unavailable.

`round_winner` is `CT` or `T`. `round_end_reason` is `bomb_defusal`,
`ct_eliminated`, `t_eliminated`, `bomb_exploded`, or `time_ran_out`.

## Fit KumoTabular

Pass a directory of prepared Parquet files or list the files individually.

```sh
uv run --group kumo cs-win-prob-train-kumo training-data/ --output kumo-context.parquet
```

The command fits the pretrained KumoTabular small classifier with eight
estimators on CUDA when available, otherwise on CPU. Override these defaults
with `--num-estimators` and `--device`. The fitter streams the input files and
keeps up to 2,048 context rows by default, with at least one example per
observed `round_end_reason`. Use `--context-rows` and `--seed` to control the
sample. Features include the map, phase, and live game-state statistics.
`round_winner` is excluded because it is only known at round end.

Kumo uses in-context learning. The output saves the labeled context rows and
fit settings, and loading it rebuilds the inference cache using the pretrained
weights. The first fit or load downloads the base checkpoint from Hugging Face.
[NVIDIA's in-context learning guide](https://nvidia.github.io/structured-data-models/icl.html)

```python
from pathlib import Path

import polars as pl

from cs_win_prob.kumo import load_kumo, predict_round_end_reason

model = load_kumo(Path("kumo-context.parquet"))
probabilities = predict_round_end_reason(model, pl.read_parquet("held-out-match.parquet"))
```

## Train and plot round win probabilities

Train directly from a directory of HLTV RAR archives. The script uses UnRAR
from `PATH` or a standard Windows WinRAR installation, including from WSL.
Use `--unrar` to point to another executable. It extracts demos into temporary
directories, parses them with Awpy, deduplicates identical demos, and fits
KumoTabular small to `round_winner` with eight estimators by default.
`--exclude-demo` excludes any archive containing the held-out file by name
or contents. Add `--states-dir artifacts/training-states` to retain the
intermediate Parquet files.

```sh
uv run --group kumo cs-win-prob-train-demos demos/ \
  --exclude-demo furia-vs-betboom-m2-mirage.dem \
  --output artifacts/round-win-context.parquet

uv run --group kumo --group plot cs-win-prob-plot-rounds \
  furia-vs-betboom-m2-mirage.dem artifacts/round-win-context.parquet \
  --output artifacts/furia-betboom-mirage-rounds-1-10.png \
  --title 'FURIA vs BetBoom · Mirage'
```

The plot scores the first 10 rounds each second and at kills, bomb plants,
and the round decision. Each panel shows the estimated CT probability and
the actual winning side. Use `--predictions` to also save the CT and T
probabilities at each scored tick as Parquet.

## Version datasets and models independently

### Release tags

All datasets describe round states. Name dataset families by representation
using `data/<representation>/vX.Y.Z`. Use `data/tabular/v1.0.0` for the tabular
dataset and `data/graph/v1.0.0` for a future graph dataset. The Kumo workflow also
accepts legacy tabular tags under `data/round-states-tabular/`.

Name models by architecture and prediction target using
`model/<architecture>-<target>/vX.Y.Z`. For example,
`model/kumo-round-end/v1.0.0` predicts the round-end reason. Use `round-win` in
the name of a model that predicts the winning side.

- **Dataset versions.** Use a minor bump for added demos or compatible columns,
  a patch for corrections, and a major bump for breaking schema changes.
- **Model versions.** Use a minor bump for a retrain with a compatible inference
  interface, a patch for packaging or metadata fixes, and a major bump for a
  breaking inference change. Give a different architecture its own model name.
  Reuse the dataset tag when only the model changes.

### Create a release

Prepare one Parquet per demo, combine them, and write a manifest for the dataset tag.

```sh
DATASET_TAG=data/tabular/v1.0.0
DATASET_DIR=artifacts/data/tabular/v1.0.0

uv run cs-win-prob-train-demos demos/ --prepare-only --states-dir "$DATASET_DIR/prepared"
uv run cs-win-prob-export-dataset "$DATASET_DIR/prepared" \
  --output "$DATASET_DIR/cs2-round-states.parquet"
uv run cs-win-prob-dataset-manifest "$DATASET_DIR/cs2-round-states.parquet" \
  --tag "$DATASET_TAG"
```

Each row includes the original `.dem` basename, its `demo_sha256`, and the
recording-local `tick`. The pair `(demo_sha256, tick)` identifies a snapshot and
can also be used by a graph export. The exporter uses a seeded 80/20 split by
demo by default. Pass `--split-from` with a previous dataset to keep its exact
demo set and split assignments. The dataset manifest records the Parquet SHA-256,
the held-out demo hashes, and a hash of the held-out snapshot keys. Regenerate
older per-demo Parquets to get the original ticks for a versioned release.

After pushing the workflow, publish the dataset release and dispatch the
model workflow with the dataset tag and the model's own version.

```sh
gh release create "$DATASET_TAG" \
  "$DATASET_DIR/cs2-round-states.parquet" "$DATASET_DIR/dataset-manifest.json" \
  --target "$(git rev-parse HEAD)" --title "$DATASET_TAG" \
  --notes 'Tabular CS2 snapshots with demo-level train and test splits.'

gh workflow run kumo-release.yml --ref main \
  -f dataset_tag="$DATASET_TAG" -f model_name=kumo-round-end -f model_version=1.0.0
```

The workflow verifies the dataset manifest, fits KumoTabular small to
`round_end_reason`, and evaluates accuracy, log loss, and multiclass Brier score
on held-out demos. After success, it publishes `model/kumo-round-end/v1.0.0`
with `round-end-reason-context.parquet`, `metrics.json`, and `model-manifest.json`.
The model manifest records the dataset tag and SHA-256, held-out split, training
commit, and checksums for the model and metrics files.

Run the same fit and evaluation locally with:

```sh
uv run --locked --group kumo cs-win-prob-fit-release artifacts/cs2-round-states.parquet \
  --output-dir artifacts/release-model
```

Add `--baseline-features` to compare with the original feature set on the same
dataset and split. Pass `--target round_winner` to fit the binary CT/T model.
