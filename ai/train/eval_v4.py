"""
TITAN V4 — Quick Evaluation Script
====================================

Runs N games of the checkpoint against several opponents and prints
a win-rate table.  Useful for tracking model progress during training.

Opponents tested:
  random    — uniformly-random legal action
  heuristic — rule-based (attack hero > attack unit > play card > end turn)
  self      — model plays against itself (mirror match)

Usage:
    # Evaluate a specific checkpoint
    python -m ai.train.eval_v4 checkpoints_v4/step_0001000.npz

    # Evaluate with more games and a fixed seed
    python -m ai.train.eval_v4 checkpoints_v4/final.npz --games 200 --seed 0

    # Compare two checkpoints (old vs new)
    python -m ai.train.eval_v4 checkpoints_v4/step_0000500.npz \
                               --vs-checkpoint checkpoints_v4/step_0001000.npz \
                               --games 100
"""

import sys
import argparse
import random as pyrandom
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn

from ai.arena_env import ArenaEnv, OBS_SIZE, TOTAL_ACTIONS
from core.state import GameStatus
from train_v4 import TitanNet, heuristic_action


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def greedy_action(net: TitanNet, obs: np.ndarray, mask: np.ndarray) -> int:
    obs_mx = mx.array(obs.reshape(1, -1))
    logits, _ = net(obs_mx)
    logits_np = np.array(logits[0])
    logits_np[mask == 0] = -1e9
    return int(np.argmax(logits_np))


def sample_action(net: TitanNet, obs: np.ndarray, mask: np.ndarray,
                  temperature: float = 1.0) -> int:
    obs_mx = mx.array(obs.reshape(1, -1))
    logits, _ = net(obs_mx)
    logits_np = np.array(logits[0])
    logits_np[mask == 0] = -1e9
    logits_np -= logits_np.max()
    probs = np.exp(logits_np / temperature)
    probs /= probs.sum() + 1e-8
    return int(np.random.choice(TOTAL_ACTIONS, p=probs))


def random_action(mask: np.ndarray) -> int:
    legal = np.where(mask > 0)[0]
    return int(np.random.choice(legal)) if len(legal) else 0


def _get_flipped_obs_and_mask(env: ArenaEnv):
    """Opponent's perspective obs + mask."""
    orig = env.agent_id
    st = env.engine.state
    env.agent_id = st.p2.user_id if orig == st.p1.user_id else st.p1.user_id
    obs  = env._get_obs()
    mask = env.get_action_mask()
    env.agent_id = orig
    return obs, mask


# ---------------------------------------------------------------------------
# Run a single evaluation match
# ---------------------------------------------------------------------------

def run_game(env: ArenaEnv, agent: TitanNet, opp_type: str,
             opp_net: TitanNet | None = None,
             greedy: bool = True) -> dict:
    """
    Returns dict with: win (bool), turns, my_hp, opp_hp
    """
    obs, _ = env.reset()
    done = False
    max_steps = 500

    for _ in range(max_steps):
        if done:
            break

        st = env.engine.state
        is_my_turn = st.current_turn_owner_id == env.agent_id

        if is_my_turn:
            mask = env.get_action_mask()
            if greedy:
                action = greedy_action(agent, obs, mask)
            else:
                action = sample_action(agent, obs, mask)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            opp_obs, opp_mask = _get_flipped_obs_and_mask(env)
            if opp_type == "random":
                action = random_action(opp_mask)
            elif opp_type == "heuristic":
                action = heuristic_action(opp_mask)
            elif opp_type in ("self", "checkpoint"):
                assert opp_net is not None
                if greedy:
                    action = greedy_action(opp_net, opp_obs, opp_mask)
                else:
                    action = sample_action(opp_net, opp_obs, opp_mask)
            else:
                raise ValueError(f"Unknown opp_type: {opp_type}")
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    st = env.engine.state
    win = (
        (st.status == GameStatus.P1_WIN and env.agent_id == st.p1.user_id)
        or (st.status == GameStatus.P2_WIN and env.agent_id == st.p2.user_id)
    )
    my_hp  = st.p1.hero.hp if env.agent_id == st.p1.user_id else st.p2.hero.hp
    opp_hp = st.p2.hero.hp if env.agent_id == st.p1.user_id else st.p1.hero.hp
    return {"win": win, "turns": st.turn_number, "my_hp": my_hp, "opp_hp": opp_hp}


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: str,
    n_games: int = 100,
    seed: int | None = None,
    vs_checkpoint: str | None = None,
    greedy: bool = True,
) -> None:
    if seed is not None:
        np.random.seed(seed)
        pyrandom.seed(seed)

    # Load agent
    raw = mx.load(checkpoint)
    agent = TitanNet()
    agent.load_weights([(k, mx.array(v)) for k, v in raw.items()])
    mx.eval(agent.parameters())
    print(f"\nAgent checkpoint : {Path(checkpoint).name}")

    # Load optional second checkpoint
    opp_net = None
    if vs_checkpoint:
        raw2 = mx.load(vs_checkpoint)
        opp_net = TitanNet()
        opp_net.load_weights([(k, mx.array(v)) for k, v in raw2.items()])
        mx.eval(opp_net.parameters())
        print(f"Opponent checkpoint: {Path(vs_checkpoint).name}")

    env = ArenaEnv()

    suites = [
        ("random",     None),
        ("heuristic",  None),
        ("self",       agent),   # mirror match
    ]
    if vs_checkpoint and opp_net is not None:
        suites.append(("checkpoint", opp_net))

    results = {}
    for opp_name, opp_net_arg in suites:
        wins, hp_diff_total, turns_total = 0, 0.0, 0
        for _ in range(n_games):
            r = run_game(env, agent, opp_name, opp_net=opp_net_arg, greedy=greedy)
            wins += int(r["win"])
            hp_diff_total += r["my_hp"] - r["opp_hp"]
            turns_total += r["turns"]
        results[opp_name] = {
            "wr":       wins / n_games,
            "hp_diff":  hp_diff_total / n_games,
            "avg_turns": turns_total / n_games,
        }

    # Pretty print
    col_w = [12, 8, 10, 10]
    header = f"{'Opponent':<{col_w[0]}}{'WR':>{col_w[1]}}{'ΔHP (avg)':>{col_w[2]}}{'Avg turns':>{col_w[3]}}"
    sep = "-" * sum(col_w)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for opp_name, r in results.items():
        wr_str = f"{r['wr']:.1%}"
        hp_str = f"{r['hp_diff']:+.1f}"
        t_str  = f"{r['avg_turns']:.1f}"
        print(f"{opp_name:<{col_w[0]}}{wr_str:>{col_w[1]}}{hp_str:>{col_w[2]}}{t_str:>{col_w[3]}}")
    print(sep)
    print(f"({n_games} games per opponent, {'greedy' if greedy else 'sampled'} policy)\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TitanNet V4 checkpoint")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per opponent (default: 100)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vs-checkpoint", type=str, default=None,
                        help="Second checkpoint to evaluate head-to-head")
    parser.add_argument("--sample", action="store_true",
                        help="Use sampled policy instead of greedy")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        n_games=args.games,
        seed=args.seed,
        vs_checkpoint=args.vs_checkpoint,
        greedy=not args.sample,
    )
