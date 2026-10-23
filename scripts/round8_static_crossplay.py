#!/usr/bin/env python3
"""Cross-play of *existing* deterministic heuristic agents in the official
Overcooked-V2 environment. Research extension, NOT an original ICRL baseline.

Ego=agent_0; teammate=agent_1 loaded from the official test manifest.
All partner theta parameters are kept exactly as the test manifest specifies.
"""
import argparse
import csv
import json
import math
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.manifest_schema import load_manifest
# Running as python scripts/<name>.py places scripts/ first on sys.path;
# explicitly add the checked-out repository root for the top-level eval module.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_icrl import create_env_for_task, create_teammate, get_manifest_path
from teammate_wrapper import make_teammate
from teammate_wrapper.specs import HeuristicTeammateSpec

HEADERS = [
    "seed", "layout", "partner_family", "task_id", "theta_id", "ego_family",
    "episode", "return", "steps", "success", "max_steps"
]

def evaluate(task, ego_family, seed, episodes, max_steps):
    env = create_env_for_task(task, max_steps=max_steps)
    partner = create_teammate(task, env)
    ego = make_teammate(
        HeuristicTeammateSpec(
            family=ego_family,
            theta="default",
            use_log_wrapper=True,
            start_cooking_interaction=False,
        ),
        env,
        "agent_0",
    )
    for ep in range(episodes):
        # Same task, seed and episode give the same environment reset key
        # across ego policies; no hidden resampling of the partner parameters.
        root_int = (
            zlib.crc32(task.task_id.encode("utf-8"))
            ^ (seed * 2654435761)
            ^ (ep * 2246822519)
        ) & 0xFFFFFFFF
        root = jax.random.PRNGKey(root_int)
        obs, state = env.reset(jax.random.fold_in(root, 0))
        carry0 = ego.init(jax.random.fold_in(root, 1))
        carry1 = partner.init(jax.random.fold_in(root, 2))
        done = {k: False for k in env.agents + ["__all__"]}
        cumulative = 0.0
        steps = 0
        for step in range(max_steps):
            key = jax.random.fold_in(root, 1000 + step)
            available = env.get_avail_actions(state.env_state)
            carry0, a0 = ego.act(
                carry0, obs["agent_0"], jnp.asarray(done["agent_0"]),
                jax.random.fold_in(key, 0),
                env_state=state, avail_actions=available["agent_0"],
            )
            carry1, a1 = partner.act(
                carry1, obs["agent_1"], jnp.asarray(done["agent_1"]),
                jax.random.fold_in(key, 1),
                env_state=state, avail_actions=available["agent_1"],
            )
            a0 = int(jnp.asarray(a0).squeeze())
            a1 = int(jnp.asarray(a1).squeeze())
            if not (0 <= a0 < 6 and 0 <= a1 < 6):
                raise RuntimeError(f"Invalid action: {a0}, {a1}")
            obs, state, reward, done, _ = env.step(
                jax.random.fold_in(key, 2), state,
                {"agent_0": a0, "agent_1": a1},
            )
            cumulative += float(reward["agent_0"])
            steps = step + 1
            if done["__all__"]:
                break
        if not (math.isfinite(cumulative) and 1 <= steps <= max_steps):
            raise RuntimeError("Invalid episode result")
        yield {
            "seed": seed,
            "layout": task.layout_name,
            "partner_family": task.teammate.family,
            "task_id": task.task_id,
            "theta_id": task.teammate.theta_id,
            "ego_family": ego_family,
            "episode": ep,
            "return": cumulative,
            "steps": steps,
            "success": int(cumulative > 0),
            "max_steps": max_steps,
        }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", default="demo_cook_simple,grounded_coord_ring")
    p.add_argument("--partners", default="assembly_line,territory,utility_greedy,recipe_aware_button")
    p.add_argument("--ego", default="recipe_aware_button,utility_greedy")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--tasks-per-cell", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--out", default="round8_crossplay.csv")
    args = p.parse_args()

    layouts = args.layouts.split(",")
    partners = args.partners.split(",")
    egos = args.ego.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for layout in layouts:
        for partner_family in partners:
            path = get_manifest_path("teammate", layout, partner_family)
            tasks = load_manifest(str(path))
            if len(tasks) < args.tasks_per_cell:
                raise ValueError(f"Insufficient manifest entries at {path}")
            for task in tasks[:args.tasks_per_cell]:
                if task.teammate.kind != "heuristic":
                    raise ValueError("Only manifest-defined heuristic teammates are supported")
                for ego_family in egos:
                    for seed in seeds:
                        for result in evaluate(task, ego_family, seed, args.episodes, args.max_steps):
                            rows.append(result)
                            print(json.dumps(result, ensure_ascii=False), flush=True)
    expected = (len(layouts) * len(partners) * args.tasks_per_cell
                * len(egos) * len(seeds) * args.episodes)
    if len(rows) != expected:
        raise AssertionError(f"Only {len(rows)} of {expected} expected episodes")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"STRICT SUCCESS: {len(rows)} of {expected} crossplay episodes -> {output}", flush=True)

if __name__ == "__main__":
    main()
