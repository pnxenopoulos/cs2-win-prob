from unittest.mock import Mock, call

import polars as pl
from awpy import Demo

from cs_win_prob.data_preparation import prepare_demo, prepare_round_timeline


def _demo() -> Mock:
    demo = Mock(spec=Demo)
    demo.header = {"map_name": "de_custom"}
    demo.parts = (demo,)
    demo.tick_rate = 64
    demo.rounds = pl.DataFrame(
        {
            "start_tick": [90, 200, 300],
            "freeze_end_tick": [100, 210, None],
            "end_tick": [103, 220, 320],
            "official_end_tick": [109, 229, 329],
            "winner": [3, 0, 2],
            "reason_name": ["bomb_defused", "draw", "terrorists_win"],
        }
    )
    demo.bomb = pl.DataFrame(
        {
            "tick": [99, 101, 102],
            "event": ["finish_plant", "start_plant", "finish_plant"],
            "bombsite": ["B", None, "A"],
            "x": [100.0, None, 0.0],
            "y": [100.0, None, 0.0],
            "z": [100.0, None, 0.0],
        }
    )
    return demo


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "side": ["terrorist"] * 5 + ["counter-terrorist"] * 2,
            "health": [24, 25, 50, 100, 0, 75, 0],
            "armor": [0, 49, 74, 75, 100, 100, 100],
            "has_helmet": [False, True, False, False, True, True, True],
            "has_defuser": [False] * 5 + [True, True],
            "primary_weapon": ["ak47", "mp5sd", None, "ssg08", "awp", "awp", "m4a1"],
            "fire_grenades": [1, 0, 0, 0, 1, 0, 1],
            "smoke_grenades": [1, 0, 0, 1, 1, 1, 1],
            "he_grenades": [0, 1, 0, 0, 1, 1, 1],
            "flashbangs": [1, 2, 0, 1, 2, 2, 2],
            "equipment_value": [3000, 2000, 200, 2500, 5000, 6000, 4000],
            "x": [3.0, 6.0, 20.0, 15.0, 0.0, 0.0, 0.0],
            "y": [4.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            "is_defusing": [False] * 5 + [True, True],
        }
    )


def test_samples_live_round_and_resets_clock_at_completed_plant() -> None:
    demo = _demo()
    players = _players()
    demo.snapshots.return_value = pl.concat(
        [players.with_columns(tick=pl.lit(tick)) for tick in range(99, 110)]
    )

    states = prepare_demo(demo, seed=42)

    assert set(demo.snapshots.call_args.kwargs["ticks"]) == {100, 101, 102, 103}
    assert states.shape == (4, 48)
    assert states["map"].to_list() == ["other"] * 4
    assert states["phase"].to_list() == ["bomb_not_planted"] * 2 + ["bomb_planted"] * 2
    assert states["time_since_phase_start"].to_list() == [0, 1 / 64, 0, 1 / 64]
    assert states["round_winner"].to_list() == ["CT"] * 4
    assert states["round_end_reason"].to_list() == ["bomb_defusal"] * 4
    assert (
        states.select(pl.col("^t_health_.*$"), pl.col("^t_armor_.*$")).row(0)
        == (1,) * 8
    )
    assert states.select("t_rifles", "t_smgs", "t_awps", "ct_rifles", "ct_awps").row(
        0
    ) == (1, 1, 0, 0, 1)
    assert states.select(
        "t_dead", "t_helmets", "ct_dead", "ct_helmets", "ct_defuse_kits"
    ).row(0) == (1, 1, 1, 1, 1)
    assert states.select("t_fire_grenades", "t_smoke_grenades", "t_he_grenades").row(
        0
    ) == (1, 2, 1)
    assert states.select("t_equipment_value", "ct_equipment_value").row(0) == (
        7700,
        6000,
    )
    assert states.select(
        "t_live_hp", "ct_live_hp", "t_flashbangs", "ct_flashbangs"
    ).row(0) == (199, 75, 4, 2)
    assert states["bombsite"].to_list() == [None, None, "A", "A"]
    assert states["defuse_in_progress"].to_list() == [False, False, True, True]
    assert states["ct_distance_to_bomb"].to_list() == [None, None, 13.0, 13.0]
    assert states["t_distance_to_bomb"].to_list() == [None, None, 5.0, 5.0]
    assert prepare_demo(demo, seed=42, include_tick=True)["tick"].to_list() == [
        100,
        101,
        102,
        103,
    ]


def test_replaces_missing_exact_tick_with_a_recorded_tick() -> None:
    demo = _demo()
    nearby = _players().with_columns(tick=pl.lit(103))
    demo.snapshots.side_effect = [nearby, nearby]

    states = prepare_demo(demo, ticks_per_round=1, seed=42, training_maps={"de_custom"})

    assert demo.snapshots.call_args_list == [
        call(ticks=[100]),
        call(start_tick=100, end_tick=103),
    ]
    assert states.select(
        "map", "phase", "time_since_phase_start", "ct_awps"
    ).rows() == [("de_custom", "bomb_planted", 1 / 64, 1)]


def test_timeline_scores_kills_plant_and_round_decision() -> None:
    demo = _demo()
    demo.rounds = demo.rounds.with_row_index("round_num", offset=1)
    demo.kills = pl.DataFrame({"tick": [99, 101, 104]})
    players = _players()
    demo.snapshots.return_value = pl.concat(
        [players.with_columns(tick=pl.lit(tick)) for tick in range(99, 110)]
    )

    states = prepare_round_timeline(demo, rounds=1)

    assert demo.snapshots.call_args == call(ticks=[100, 101, 102, 103])
    assert states["tick"].to_list() == [100, 101, 102, 103]
    assert states["round_elapsed_seconds"].to_list() == [0, 1 / 64, 2 / 64, 3 / 64]
    assert states["plant_seconds"].to_list() == [2 / 64] * 4
    assert states["time_since_phase_start"].to_list() == [0, 1 / 64, 0, 1 / 64]
