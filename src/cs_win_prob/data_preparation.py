"""Sample labeled round states from Counter-Strike 2 demos."""

import json
from bisect import bisect_left
from collections.abc import Collection
from hashlib import file_digest
from itertools import chain
from pathlib import Path
from random import Random

import polars as pl
from awpy import Demo

DEFAULT_TICKS_PER_ROUND = 4
DEMO_METADATA_KEY = "cs_win_prob.demo.v1"
DEFAULT_TRAINING_MAPS = frozenset(
    {
        "de_ancient",
        "de_anubis",
        "de_cache",
        "de_dust2",
        "de_inferno",
        "de_mirage",
        "de_nuke",
        "de_overpass",
        "de_train",
        "de_vertigo",
    }
)

_WINNERS = {2: "T", 3: "CT"}
_END_REASONS = {
    "bomb_defused": "bomb_defusal",
    "terrorists_win": "ct_eliminated",
    "ct_win": "t_eliminated",
    "target_bombed": "bomb_exploded",
    "target_saved": "time_ran_out",
}
_SIDES = {"t": "terrorist", "ct": "counter-terrorist"}
_BUCKETS = ((0, 24), (25, 49), (50, 74), (75, 100))
_RIFLES = ("ak47", "aug", "famas", "galilar", "m4a1", "m4a1_silencer", "sg556")
_SMGS = ("bizon", "mac10", "mp5sd", "mp7", "mp9", "p90", "ump45")

_SAMPLE_SCHEMA = {
    "tick": pl.Int32,
    "_plant_tick": pl.Int32,
    "map": pl.String,
    "phase": pl.String,
    "time_since_phase_start": pl.Float64,
    "round_winner": pl.String,
    "round_end_reason": pl.String,
}


def _team_features() -> dict[str, pl.Expr]:
    health = pl.col("health")
    armor = pl.col("armor")
    primary = pl.col("primary_weapon")
    features = {}
    for prefix, side in _SIDES.items():
        on_side = pl.col("side") == side
        alive = on_side & (health > 0)
        features[f"{prefix}_dead"] = on_side & (health <= 0)
        for low, high in _BUCKETS:
            features[f"{prefix}_health_{low}_{high}"] = alive & health.is_between(
                low, high
            )
            features[f"{prefix}_armor_{low}_{high}"] = alive & armor.is_between(
                low, high
            )
        features[f"{prefix}_helmets"] = alive & pl.col("has_helmet")
        for item in (
            "fire_grenades",
            "smoke_grenades",
            "he_grenades",
            "equipment_value",
        ):
            features[f"{prefix}_{item}"] = pl.col(item).filter(alive)
        features[f"{prefix}_rifles"] = alive & primary.is_in(_RIFLES)
        features[f"{prefix}_smgs"] = alive & primary.is_in(_SMGS)
        features[f"{prefix}_awps"] = alive & (primary == "awp")
    features["ct_defuse_kits"] = (
        (pl.col("side") == _SIDES["ct"]) & (health > 0) & pl.col("has_defuser")
    )
    return {name: value.cast(pl.Int64).sum() for name, value in features.items()}


_TEAM_FEATURES = _team_features()
BASE_FEATURE_COLUMNS = ("map", "phase", "time_since_phase_start", *_TEAM_FEATURES)
_BOMB_COORDINATES = ("_bomb_x", "_bomb_y", "_bomb_z")
_EXTRA_FEATURE_SCHEMA = {
    "t_live_hp": pl.Int64,
    "ct_live_hp": pl.Int64,
    "t_flashbangs": pl.Int64,
    "ct_flashbangs": pl.Int64,
    "bombsite": pl.String,
    "defuse_in_progress": pl.Boolean,
    "ct_distance_to_bomb": pl.Float64,
    "t_distance_to_bomb": pl.Float64,
}
FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *_EXTRA_FEATURE_SCHEMA)
ROUND_END_REASONS = frozenset(_END_REASONS.values())
_COMPETITIVE_SIDE = pl.col("side").is_in(list(_SIDES.values()))


def available_feature_columns(columns: Collection[str]) -> tuple[str, ...]:
    """Return the known feature columns in model order, including the required base schema."""
    return (
        *BASE_FEATURE_COLUMNS,
        *(name for name in _EXTRA_FEATURE_SCHEMA if name in columns),
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def demo_metadata(path: Path, sha256: str) -> dict[str, str]:
    return {DEMO_METADATA_KEY: json.dumps({"filename": path.name, "sha256": sha256})}


def _playable_rounds(rounds: pl.DataFrame) -> pl.DataFrame:
    return rounds.filter(
        pl.col("freeze_end_tick") <= pl.col("end_tick"),
        pl.col("winner").is_in(list(_WINNERS)),
        pl.col("reason_name").is_in(list(_END_REASONS)),
    ).sort("freeze_end_tick")


def _map_name(demo: Demo, training_maps: Collection[str]) -> str:
    name = demo.header.get("map_name")
    return name if isinstance(name, str) and name in training_maps else "other"


def _plants(demo: Demo) -> pl.DataFrame:
    # Awpy reports the planter's position at plant completion, a proxy for the C4 position.
    return (
        demo.bomb.filter(pl.col("event") == "finish_plant")
        .select(
            pl.col("tick").alias("_plant_tick"),
            "bombsite",
            *(pl.col(axis).cast(pl.Float64).alias(f"_bomb_{axis}") for axis in "xyz"),
        )
        .unique(subset="_plant_tick", keep="first", maintain_order=True)
        .sort("_plant_tick")
    )


def _round_plant(plants: list[int], start: int, end: int) -> int | None:
    index = bisect_left(plants, start)
    return plants[index] if index < len(plants) and plants[index] <= end else None


def _phase(
    tick: int, start: int, plant: int | None, tick_rate: float
) -> tuple[str, float]:
    if plant is not None and tick >= plant:
        return "bomb_planted", (tick - plant) / tick_rate
    return "bomb_not_planted", (tick - start) / tick_rate


def _aggregate_samples(
    samples: pl.DataFrame, snapshots: pl.DataFrame, plants: pl.DataFrame
) -> pl.DataFrame:
    states = samples.join(plants, on="_plant_tick", how="left", maintain_order="left")
    players = snapshots.join(
        states.select("tick", "_plant_tick", *_BOMB_COORDINATES), on="tick", how="inner"
    )
    squared_distances = [
        (pl.col(axis) - pl.col(f"_bomb_{axis}")) ** 2 for axis in "xyz"
    ]
    distance = pl.sum_horizontal(*squared_distances, ignore_nulls=False).sqrt()
    live = pl.col("health") > 0
    ct = live & (pl.col("side") == _SIDES["ct"])
    t = live & (pl.col("side") == _SIDES["t"])
    counts = players.group_by("tick").agg(
        **_TEAM_FEATURES,
        t_live_hp=pl.col("health").filter(t).cast(pl.Int64).sum(),
        ct_live_hp=pl.col("health").filter(ct).cast(pl.Int64).sum(),
        t_flashbangs=pl.col("flashbangs").filter(t).cast(pl.Int64).sum(),
        ct_flashbangs=pl.col("flashbangs").filter(ct).cast(pl.Int64).sum(),
        defuse_in_progress=(
            ct & pl.col("_plant_tick").is_not_null() & pl.col("is_defusing")
        ).any(),
        ct_distance_to_bomb=distance.filter(ct).min(),
        t_distance_to_bomb=distance.filter(t).min(),
    )
    return states.drop("_plant_tick", *_BOMB_COORDINATES).join(
        counts, on="tick", how="inner", maintain_order="left"
    )


def _prepare_recording(
    demo: Demo, map_name: str, ticks_per_round: int, rng: Random, *, include_tick: bool
) -> pl.DataFrame:
    rounds = _playable_rounds(demo.rounds).select(
        "freeze_end_tick", "end_tick", "winner", "reason_name"
    )
    output_columns = (*FEATURE_COLUMNS, "round_winner", "round_end_reason")
    if include_tick:
        output_columns = ("tick", *output_columns)
    if rounds.is_empty():
        schema = (
            _SAMPLE_SCHEMA
            | dict.fromkeys(_TEAM_FEATURES, pl.Int64)
            | _EXTRA_FEATURE_SCHEMA
        )
        return pl.DataFrame(schema=schema).select(output_columns)

    picks = [
        rng.sample(range(start, end + 1), min(ticks_per_round, end - start + 1))
        for start, end in rounds.select("freeze_end_tick", "end_tick").iter_rows()
    ]
    requested = list(chain.from_iterable(picks))
    snapshots = demo.snapshots(ticks=requested).filter(
        _COMPETITIVE_SIDE, pl.col("tick").is_in(requested)
    )
    frames = [snapshots]
    observed = set(snapshots["tick"])

    plants = _plants(demo)
    plant_ticks = plants["_plant_tick"].to_list()
    tick_rate = demo.tick_rate
    rows = []
    for (start, end, winner, reason), sampled in zip(
        rounds.iter_rows(), picks, strict=True
    ):
        ticks = observed.intersection(sampled)
        if len(ticks) < len(sampled):
            # Some recordings omit exact ticks. Fill the gap from recorded states in the round.
            window = demo.snapshots(start_tick=start, end_tick=end).filter(
                _COMPETITIVE_SIDE
            )
            candidates = sorted(set(window["tick"]) - ticks)
            additional = rng.sample(
                candidates, min(len(sampled) - len(ticks), len(candidates))
            )
            frames.append(window.filter(pl.col("tick").is_in(additional)))
            ticks.update(additional)

        plant = _round_plant(plant_ticks, start, end)
        for tick in sorted(ticks):
            phase, elapsed = _phase(tick, start, plant, tick_rate)
            rows.append(
                (
                    tick,
                    plant if phase == "bomb_planted" else None,
                    map_name,
                    phase,
                    elapsed,
                    _WINNERS[winner],
                    _END_REASONS[reason],
                )
            )

    samples = pl.DataFrame(rows, schema=_SAMPLE_SCHEMA, orient="row")
    return _aggregate_samples(samples, pl.concat(frames), plants).select(output_columns)


def prepare_demo(
    demo: Demo,
    *,
    ticks_per_round: int = DEFAULT_TICKS_PER_ROUND,
    seed: int | None = None,
    training_maps: Collection[str] = DEFAULT_TRAINING_MAPS,
    include_tick: bool = False,
) -> pl.DataFrame:
    """Return one labeled row per sampled game state, in recording order.

    Sample distinct ticks from freeze end through the round decision, inclusive.
    Rounds without a playable interval or a supported outcome are skipped. Short
    or sparse rounds contribute up to the number of available snapshots.
    """
    if ticks_per_round < 1:
        raise ValueError("ticks_per_round must be at least 1")

    map_name = _map_name(demo, training_maps)
    rng = Random(seed)
    return pl.concat(
        [
            _prepare_recording(
                part, map_name, ticks_per_round, rng, include_tick=include_tick
            )
            for part in demo.parts
        ]
    )


def prepare_round_timeline(
    demo: Demo,
    *,
    rounds: int = 10,
    training_maps: Collection[str] = DEFAULT_TRAINING_MAPS,
) -> pl.DataFrame:
    """Sample the first rounds of one recording each second and at kills and plants."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    selected = _playable_rounds(demo.rounds.sort("round_num").head(rounds)).select(
        "round_num", "freeze_end_tick", "end_tick", "winner"
    )
    if selected.is_empty():
        raise ValueError("no playable rounds to plot")
    name = _map_name(demo, training_maps)
    tick_rate = demo.tick_rate
    step = max(1, round(tick_rate))
    plants = _plants(demo)
    plant_ticks = plants["_plant_tick"].to_list()
    kills = sorted(demo.kills["tick"])

    rows = []
    for number, start, end, winner in selected.iter_rows():
        plant = _round_plant(plant_ticks, start, end)
        plant_seconds = (plant - start) / tick_rate if plant is not None else None
        ticks = set(range(start, end + 1, step))
        ticks.update(kills[bisect_left(kills, start) : bisect_left(kills, end + 1)])
        ticks.add(end)
        if plant is not None:
            ticks.add(plant)
        for tick in sorted(ticks):
            phase, elapsed = _phase(tick, start, plant, tick_rate)
            rows.append(
                (
                    tick,
                    plant if phase == "bomb_planted" else None,
                    number,
                    (tick - start) / tick_rate,
                    (end - start) / tick_rate,
                    plant_seconds,
                    name,
                    phase,
                    elapsed,
                    _WINNERS[winner],
                )
            )

    samples = pl.DataFrame(
        rows,
        schema={
            "tick": pl.Int32,
            "_plant_tick": pl.Int32,
            "round_num": pl.Int32,
            "round_elapsed_seconds": pl.Float64,
            "round_duration_seconds": pl.Float64,
            "plant_seconds": pl.Float64,
            "map": pl.String,
            "phase": pl.String,
            "time_since_phase_start": pl.Float64,
            "round_winner": pl.String,
        },
        orient="row",
    )
    snapshots = demo.snapshots(ticks=samples["tick"].to_list()).filter(
        _COMPETITIVE_SIDE
    )
    timeline = _aggregate_samples(samples, snapshots, plants)
    if timeline["round_num"].n_unique() != selected.height:
        raise ValueError("missing player snapshots for one or more selected rounds")
    return timeline
