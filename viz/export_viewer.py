"""Export one saved run as replay data.

    python -m viz.export_viewer --file <h5-name> --genome 0 --run 1

Re-runs a single saved run deterministically from its stored seeds and records
every tick. Experiments themselves stay headless and parallel; this is done
afterwards, only for runs worth looking at.

The replay is driven by `mvb.simulation_API.simulate_run`, the same tick loop the
experiment used, so it cannot drift from the simulation it reproduces.

Writes one self-contained .html: CSS, JS and the run data are all inlined, so
it opens straight from disk with no server and no network access.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mvb.world import World
from mvb.worm import Worm
from mvb.feeding import seed_food
from mvb.simulation_API import simulate_run
from mvb.simulation_helper_functions import load_brain_module
from viz.replay_recorder import WorldFrameRecorder
from viz.run_source import RunSource, load_run_source, resolve_h5_path, summarize_file
from viz.verify import verify, format_report


# ============================================================
# Replay
# ============================================================

def replay(source: RunSource, max_ticks: int = None) -> WorldFrameRecorder:
    """Re-run one recorded run with a recorder attached.

    Mirrors the per-run setup in `eval_variant` so the replayed run sees exactly
    the same world, genome and rng streams as the original.

    Args:
        source: RunSource describing which run to replay
        max_ticks: Override the tick limit (defaults to the experiment's)

    Returns:
        The WorldFrameRecorder holding the recorded frames.
    """
    cfg = source.cfg
    world_cfg, worm_cfg, brain_cfg = cfg["world"], cfg["worm"], cfg["brain"]

    phases = sorted(cfg["food"], key=lambda p: p["phase_from"])
    feeding_cfg, switch_phases = phases[0], phases[1:]

    brain_module = load_brain_module(str(worm_cfg["decisionmaking"]["version"]))
    # Never attach a Qt brain window during export
    brain_module._brain_renderer = None

    world = World(
        world_cfg["grid_width"],
        world_cfg["grid_height"],
        tuple(world_cfg["start_pos"]),
        0,
    )
    world.feeding_cfg = feeding_cfg

    worm = Worm(
        worm_cfg["speed"],
        worm_cfg["energy_capacity"],
        worm_cfg["metabolic_rate"],
        worm_cfg["movement_cost"],
        world,
    )
    worm.active_sensors = worm_cfg.get("sensors", {}).get("active", ["current_field"])
    worm.brain = brain_module

    # Same order as eval_variant: init brain, then reset world and worm
    brain_module.init_brain(source.genome, brain_cfg)

    world.rng_world_run = np.random.default_rng(source.world_seed)
    world.switch_phases = list(switch_phases)
    world.reset_food()
    seed_food(world, feeding_cfg)
    worm.reset()

    recorder = WorldFrameRecorder(world, worm)
    recorder.start()
    worm.renderer = recorder

    simulate_run(
        world,
        worm,
        None,                                        # no MetricsRecorder needed
        np.random.default_rng(source.decision_seed),
        np.random.default_rng(source.noise_seed),
        max_ticks if max_ticks is not None else source.max_ticks,
        None,                                        # no pause manager
    )
    return recorder


# ============================================================
# Page building
# ============================================================

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"


def _embed_json(payload: dict) -> str:
    """Serialise run data for inlining inside a <script> tag.

    `</script>` appearing anywhere inside the JSON would close the tag early and
    break the page, so `<` is escaped. It stays valid JSON either way.
    """
    return json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")


def build_page(payload: dict, title: str, meta: str) -> str:
    """Inline the template's CSS, JS and data into a single HTML document."""
    html = (TEMPLATE_DIR / "viewer.html").read_text(encoding="utf-8")
    css = (TEMPLATE_DIR / "viewer.css").read_text(encoding="utf-8")
    js = (TEMPLATE_DIR / "viewer.js").read_text(encoding="utf-8")

    # Substituted rather than .format()ed: the CSS and JS are full of braces.
    for token, value in (
        ("__TITLE__", title),
        ("__META__", meta),
        ("__CSS__", css),
        ("__JS__", js),
        ("__DATA__", _embed_json(payload)),
    ):
        html = html.replace(token, value)

    leftover = [t for t in ("__TITLE__", "__META__", "__CSS__", "__JS__", "__DATA__") if t in html]
    if leftover:
        raise RuntimeError(f"[ERROR] template placeholders not substituted: {leftover}")
    return html


# ============================================================
# CLI
# ============================================================

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a saved Byte run as replay data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m viz.export_viewer --file 2026-09-16_01-41-48_test_batch2 --genome 0 --run 1
  python -m viz.export_viewer --file data/temp/my_run.h5 --genome 2 --run 0 --open
        """,
    )
    parser.add_argument("--file", required=True,
                        help="HDF5 result file (path, or bare name to find under data/)")
    parser.add_argument("--genome", type=int, default=0,
                        help="Elite id (EA output) or variant id (batch output). Default 0")
    parser.add_argument("--run", type=int, default=0,
                        help="Which run of that genome to replay. Default 0")
    parser.add_argument("--out", default=None,
                        help="Output path. Defaults to <h5-stem>_g<genome>_r<run>.html beside the source")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Override the recorded tick limit")
    parser.add_argument("--json", action="store_true",
                        help="Also write the raw recording as .json next to the page")
    parser.add_argument("--list", action="store_true",
                        help="List the genomes and runs in the file, then exit")
    parser.add_argument("--verify", action="store_true",
                        help="Compare the replay tick-by-tick against the file's per_tick data")
    parser.add_argument("--open", action="store_true",
                        help="Open the exported page in the default browser")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)

    try:
        hdf5_path = resolve_h5_path(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1

    if args.list:
        print(summarize_file(hdf5_path))
        return 0

    try:
        source = load_run_source(hdf5_path, args.genome, args.run)
    except (ValueError, KeyError) as e:
        print(e)
        return 1

    print(f"[export] {hdf5_path}")
    print(f"[export] layout={source.layout}  genome={source.genome_id}  run={source.run_id}")
    print(f"[export] available genomes={source.available_genomes}  runs={source.available_runs}")
    print(f"[export] seeds  world={source.world_seed}  noise={source.noise_seed}  "
          f"decision={source.decision_seed}")

    recorder = replay(source, max_ticks=args.max_ticks)
    print(f"[export] {recorder.summary()}")

    # The replay should live exactly as long as the original. A mismatch means
    # it diverged, and nothing downstream can be trusted. --verify compares every
    # tick rather than just the total.
    if source.expected_lifetime >= 0:
        actual = recorder.frames[-1]["t"]
        match = "OK" if actual == source.expected_lifetime else "MISMATCH"
        print(f"[export] lifetime recorded={source.expected_lifetime} "
              f"replayed={actual}  [{match}]")
        if actual != source.expected_lifetime:
            print("[export] WARNING: replay diverged from the original run.")

    verify_failed = False
    if args.verify:
        checks, fatal = verify(recorder, source)
        print()
        print(format_report(checks, fatal, source))
        print()
        verify_failed = not fatal and any(
            not c.ok for c in checks if c.name != "movement"
        )

    stem = f"{Path(hdf5_path).stem}_g{source.genome_id}_r{source.run_id}"
    out_path = Path(args.out) if args.out else Path(hdf5_path).with_name(f"{stem}.html")

    payload = recorder.to_dict()
    payload["meta"] = {
        "source_file": str(hdf5_path),
        "layout": source.layout,
        "genome_id": source.genome_id,
        "run_id": source.run_id,
        "world_seed": source.world_seed,
        "noise_seed": source.noise_seed,
        "decision_seed": source.decision_seed,
    }

    final_tick = recorder.frames[-1]["t"]
    title = f"Byte replay · {Path(hdf5_path).stem} · g{source.genome_id} r{source.run_id}"
    meta = (f"{Path(hdf5_path).name} &nbsp;·&nbsp; genome {source.genome_id} &nbsp;·&nbsp; "
            f"run {source.run_id} &nbsp;·&nbsp; {final_tick} ticks &nbsp;·&nbsp; "
            f"seed {source.world_seed}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_page(payload, title, meta), encoding="utf-8")
    print(f"[export] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    if args.json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"[export] wrote {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")

    print(f"[export] open it: {out_path.resolve()}")

    if args.open:
        import webbrowser
        webbrowser.open(out_path.resolve().as_uri())

    return 1 if verify_failed else 0


if __name__ == "__main__":
    sys.exit(main())
