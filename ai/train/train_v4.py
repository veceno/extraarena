"""
TITAN V4 — M4 Pro Optimized PPO with Curriculum Learning
=========================================================

Improvements over V3:
  - 12 CPU workers  (M4 Pro: 10 P-cores + 4 E-cores)
  - MLX Metal GPU for all model updates (mx.set_default_device(mx.gpu))
  - Curriculum: random → heuristic → self-play → league
  - Fixed GAE: proper per-step done flags (was broken in V3)
  - Fixed PPO loss: mx broadcast one-hot instead of broken indexed assign
  - 4 PPO epochs per update batch
  - Entropy bonus to prevent premature convergence
  - League pool of up to 20 historical checkpoints
  - Per-worker command queues (no race conditions)
  - Opponent always sees state from its own perspective

Training phases (by update count):
    0   – 100  : vs Random agent      → learn basic game rules
  100   – 500  : vs Heuristic agent   → learn basic strategy
  500   – 3000 : Self-play            → refine strategy
  3000+        : Self + League (50%)  → robustness, no forgetting

Usage:
    cd extraarena
    python -m ai.train.train_v4
    python -m ai.train.train_v4 --resume checkpoints_v4/step_000500.npz
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import traceback

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.arena_env import ArenaEnv, OBS_SIZE, TOTAL_ACTIONS
from core.state import GameStatus

# ---------------------------------------------------------------------------
# M4 Pro config
# ---------------------------------------------------------------------------
HIDDEN_DIM = 512
NUM_LAYERS = 5
LEARNING_RATE = 3e-4
LR_MIN      = 3e-5       # floor for cosine LR decay (league phase)

NUM_ACTORS = 12           # M4 Pro: 10 P + 4 E cores
STEPS_PER_ACTOR = 256
BATCH_SIZE = NUM_ACTORS * STEPS_PER_ACTOR   # 3 072

PPO_EPOCHS = 4
MINI_BATCH = 512
CLIP_EPS = 0.2
# ENTROPY_COEF is now adaptive — see get_entropy_coef()
VALUE_COEF = 0.5
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_TURNS = 60

# Opponent temperature during self-play / league (prevents deterministic exploit)
OPP_TEMPERATURE = 0.8

# Curriculum thresholds (update count)
PHASE_RANDOM_END    = 150   # ~460 K steps — stabilise vs random
PHASE_HEURISTIC_END = 800   # ~650 K steps — consistently beat heuristic
PHASE_SELF_END      = 5000  # ~13 M steps  — main self-play phase

LEAGUE_POOL_SIZE = 30
CHECKPOINT_EVERY = 25       # finer granularity → richer league pool
LEAGUE_PROB = 0.4           # league phase: 60 % self + 40 % historical

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints_v4"
LEAGUE_DIR = BASE_DIR / "league_pool"


# ---------------------------------------------------------------------------
# Model  (architecture unchanged → ONNX-exportable)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.norm(self.fc(x))) + x


class TitanNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_SIZE, act_dim: int = TOTAL_ACTIONS):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, HIDDEN_DIM)
        self.blocks = [ResBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)]
        self.actor_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, act_dim)
        )
        self.critic_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 1)
        )

    def __call__(self, x: mx.array):
        x = nn.silu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.actor_head(x), self.critic_head(x)


# ---------------------------------------------------------------------------
# Helpers shared between main process and workers
# ---------------------------------------------------------------------------
def heuristic_action(mask: np.ndarray) -> int:
    """
    Simple rule-based policy derived purely from the action mask.
    Priority: attack hero > attack unit > play card > end turn.
    """
    MAX_HAND = 4
    MAX_BOARD = 5
    base_atk = 1 + MAX_HAND * 17  # = 69

    # 1. Attack opponent hero
    for i in range(MAX_BOARD):
        idx = base_atk + i * 8 + 7
        if mask[idx] > 0:
            return idx

    # 2. Attack opponent units
    for i in range(MAX_BOARD):
        for j in range(7):
            idx = base_atk + i * 8 + j
            if mask[idx] > 0:
                return idx

    # 3. Play a card (first available)
    for i in range(MAX_HAND * 17):
        idx = 1 + i
        if mask[idx] > 0:
            return idx

    return 0  # end turn


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_val: float,
    gamma: float = GAE_GAMMA,
    lam: float = GAE_LAMBDA,
):
    """
    Fixed GAE.  'dones' is a per-step float array (1.0 = terminal step).
    V3 passed a single bool; this caused wrong advantages at episode ends.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        non_terminal = 1.0 - dones[t]
        next_val = last_val if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        advantages[t] = gae = delta + gamma * lam * non_terminal * gae
    returns = advantages + values
    return advantages, returns


def params_to_numpy(model: TitanNet) -> dict:
    return {k: np.array(v) for k, v in dict(nn.utils.tree_flatten(model.parameters())).items()}


def load_params(model: TitanNet, params: dict):
    model.load_weights([(k, mx.array(v)) for k, v in params.items()])


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _get_opp_obs(env: ArenaEnv) -> np.ndarray:
    """Observation from the opponent's point of view."""
    orig = env.agent_id
    st = env.engine.state
    env.agent_id = st.p2.user_id if orig == st.p1.user_id else st.p1.user_id
    obs = env._get_obs()
    env.agent_id = orig
    return obs


def _get_opp_mask(env: ArenaEnv) -> np.ndarray:
    """Action mask from the opponent's point of view."""
    orig = env.agent_id
    st = env.engine.state
    env.agent_id = st.p2.user_id if orig == st.p1.user_id else st.p1.user_id
    mask = env.get_action_mask()
    env.agent_id = orig
    return mask


def worker_fn(
    worker_id: int,
    cmd_queue: mp.Queue,
    traj_queue: mp.Queue,
    cancel_event: mp.Event,
):
    """
    Pure-CPU worker: simulates full episodes and sends trajectory dicts
    to the main process.  No Metal GPU calls here.
    """
    try:
        env = ArenaEnv()
        local_net = TitanNet()
        opp_net = TitanNet()
        opp_type = "random"

        # Wait for the first weight broadcast before starting
        while not cancel_event.is_set():
            try:
                msg = cmd_queue.get(timeout=5.0)
                local_params, opp_t, opp_params = msg
                load_params(local_net, local_params)
                if opp_params is not None:
                    load_params(opp_net, opp_params)
                opp_type = opp_t
                break
            except Empty:
                continue

        while not cancel_event.is_set():
            # Drain stale weight updates (keep the newest)
            new_msg = None
            while not cmd_queue.empty():
                try:
                    new_msg = cmd_queue.get_nowait()
                except Empty:
                    break
            if new_msg is not None:
                local_params, opp_t, opp_params = new_msg
                load_params(local_net, local_params)
                if opp_params is not None:
                    load_params(opp_net, opp_params)
                opp_type = opp_t

            obs, _ = env.reset()
            states, actions, rewards, values, log_probs, dones_arr, masks = (
                [], [], [], [], [], [], []
            )
            done = False
            step = 0

            while not done and step < STEPS_PER_ACTOR:
                st = env.engine.state
                is_my_turn = st.current_turn_owner_id == env.agent_id

                if is_my_turn:
                    obs_mx = mx.array(obs.reshape(1, -1))
                    mask_np = env.get_action_mask()
                    logits, val = local_net(obs_mx)
                    logits_np = np.array(logits[0])
                    logits_np[mask_np == 0] = -1e9
                    logits_np -= logits_np.max()
                    probs = np.exp(logits_np)
                    probs /= probs.sum() + 1e-8

                    action = int(np.random.choice(TOTAL_ACTIONS, p=probs))
                    log_prob = float(np.log(probs[action] + 1e-10))

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    states.append(obs)
                    actions.append(action)
                    rewards.append(float(reward))
                    values.append(float(val.item()))
                    log_probs.append(log_prob)
                    dones_arr.append(float(done))
                    masks.append(mask_np)

                    obs = next_obs
                    step += 1
                else:
                    # Opponent's turn — use opponent's own perspective
                    opp_obs = _get_opp_obs(env)
                    opp_mask = _get_opp_mask(env)

                    if opp_type == "random":
                        legal = np.where(opp_mask > 0)[0]
                        opp_action = int(np.random.choice(legal)) if len(legal) else 0
                    elif opp_type == "heuristic":
                        opp_action = heuristic_action(opp_mask)
                    else:  # "network" (self-play or league)
                        opp_obs_mx = mx.array(opp_obs.reshape(1, -1))
                        opp_logits, _ = opp_net(opp_obs_mx)
                        opp_logits_np = np.array(opp_logits[0])
                        opp_logits_np[opp_mask == 0] = -1e9
                        # Temperature sampling keeps opponent non-deterministic →
                        # harder to overfit to one fixed strategy during self-play
                        opp_logits_np -= opp_logits_np.max()
                        opp_probs = np.exp(opp_logits_np / OPP_TEMPERATURE)
                        opp_probs /= opp_probs.sum() + 1e-8
                        opp_action = int(np.random.choice(TOTAL_ACTIONS, p=opp_probs))

                    next_obs, _, terminated, truncated, _ = env.step(opp_action)
                    obs = next_obs
                    done = terminated or truncated

            if not states:
                continue

            last_val = 0.0
            if not done:
                obs_mx = mx.array(obs.reshape(1, -1))
                _, val = local_net(obs_mx)
                last_val = float(val.item())

            st = env.engine.state
            win = int(
                (st.status == GameStatus.P1_WIN and env.agent_id == st.p1.user_id)
                or (st.status == GameStatus.P2_WIN and env.agent_id == st.p2.user_id)
            )

            traj_queue.put({
                "states":    np.array(states,    dtype=np.float32),
                "actions":   np.array(actions,   dtype=np.int32),
                "rewards":   np.array(rewards,   dtype=np.float32),
                "values":    np.array(values,    dtype=np.float32),
                "log_probs": np.array(log_probs, dtype=np.float32),
                "dones":     np.array(dones_arr, dtype=np.float32),
                "masks":     np.array(masks,     dtype=np.float32),
                "last_val":  last_val,
                "win":       win,
                "opp_type":  opp_type,
                "turns":     st.turn_number,
                "my_hp":  st.p1.hero.hp if env.agent_id == st.p1.user_id else st.p2.hero.hp,
                "opp_hp": st.p2.hero.hp if env.agent_id == st.p1.user_id else st.p1.hero.hp,
            })

    except Exception:
        print(f"[Worker {worker_id}] CRASH:")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def get_phase(update: int) -> tuple[str, str]:
    """(display_name, opp_type_tag)"""
    if update < PHASE_RANDOM_END:
        return "Random", "random"
    if update < PHASE_HEURISTIC_END:
        return "Heuristic", "heuristic"
    if update < PHASE_SELF_END:
        return "Self-Play", "network"
    return "League", "league"


def get_entropy_coef(update: int) -> float:
    """
    Adaptive entropy coefficient.
    High early (explore the game), low late (exploit learned strategy).
    """
    if update < PHASE_RANDOM_END:
        return 0.05
    if update < PHASE_HEURISTIC_END:
        return 0.03
    if update < PHASE_SELF_END:
        return 0.01
    return 0.005


def get_lr(update: int) -> float:
    """
    Cosine decay from LEARNING_RATE to LR_MIN, starting at PHASE_SELF_END.
    Flat before that (model still catching up to curriculum changes).
    """
    import math
    if update < PHASE_SELF_END:
        return LEARNING_RATE
    progress = min((update - PHASE_SELF_END) / 10_000, 1.0)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LEARNING_RATE - LR_MIN) * cosine


def train(resume: str | None = None):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LEAGUE_DIR.mkdir(parents=True, exist_ok=True)

    # Force all MLX ops onto the Metal GPU
    mx.set_default_device(mx.gpu)

    master = TitanNet()
    if resume:
        w = mx.load(resume)
        master.load_weights(list(w.items()))
        print(f"Resumed from {resume}")
    mx.eval(master.parameters())

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    # League pool: list of (Path, update_number, winrate)
    league_pool: list[tuple[Path, int, float]] = []

    # Per-worker queues  →  no shared-queue race conditions
    cmd_queues = [mp.Queue(maxsize=3) for _ in range(NUM_ACTORS)]
    traj_queue = mp.Queue(maxsize=NUM_ACTORS * 4)
    cancel_event = mp.Event()

    workers = [
        mp.Process(
            target=worker_fn,
            args=(i, cmd_queues[i], traj_queue, cancel_event),
            daemon=True,
        )
        for i in range(NUM_ACTORS)
    ]
    for w in workers:
        w.start()

    def broadcast(update: int):
        _, opp_tag = get_phase(update)
        local_w = params_to_numpy(master)
        for q in cmd_queues:
            # Flush stale updates
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
            if opp_tag == "league" and league_pool:
                actual = "league" if random.random() < LEAGUE_PROB else "network"
                if actual == "league":
                    path, _, _ = random.choice(league_pool)
                    try:
                        lw = mx.load(str(path))
                        opp_w = {k: np.array(v) for k, v in lw.items()}
                    except Exception:
                        opp_w = local_w  # fallback
                else:
                    opp_w = local_w
                q.put((local_w, actual, opp_w))
            elif opp_tag == "network":
                q.put((local_w, "network", local_w))
            else:
                q.put((local_w, opp_tag, None))

    broadcast(0)

    # PPO loss — entropy_coef passed as scalar so it can vary per update
    def ppo_loss(model, obs, acts, lp_old, returns, adv, mask, ent_coef):
        logits, vals = model(obs)
        logits = logits + (mask - 1.0) * 1e9

        log_probs_all = nn.log_softmax(logits, axis=-1)

        # Gather log-prob of taken actions via broadcast one-hot
        oh = (mx.arange(TOTAL_ACTIONS) == acts[:, None]).astype(mx.float32)
        log_pi = mx.sum(log_probs_all * oh, axis=-1)

        ratio = mx.exp(log_pi - lp_old)
        surr1 = ratio * adv
        surr2 = mx.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
        policy_loss = -mx.mean(mx.minimum(surr1, surr2))

        value_loss = VALUE_COEF * mx.mean((returns - vals.squeeze(-1)) ** 2)

        probs = nn.softmax(logits, axis=-1)
        entropy = -mx.mean(mx.sum(probs * log_probs_all, axis=-1))

        return policy_loss + value_loss - ent_coef * entropy

    loss_and_grad = nn.value_and_grad(master, ppo_loss)

    # Rich UI
    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="metrics", size=14),
        Layout(name="log", ratio=1),
    )

    stats = {
        "updates": 0,
        "games": 0,
        "wins": 0,
        "loss": 0.0,
        "lr":   LEARNING_RATE,
        "ent":  0.05,
        "phase": "Random",
        "by_opp": {k: [0, 0] for k in ("random", "heuristic", "network", "league")},
    }
    match_log: list[str] = []
    buffer: list[dict] = []

    with Live(layout, refresh_per_second=4, console=console):
        while stats["updates"] < 100_000_000:
            try:
                try:
                    traj = traj_queue.get(timeout=3.0)
                except Empty:
                    buf_steps = sum(len(t["rewards"]) for t in buffer)
                    layout["header"].update(Panel(
                        f"[yellow]WAITING FOR DATA… "
                        f"buffer={buf_steps}/{BATCH_SIZE}[/]"
                    ))
                    continue

                buffer.append(traj)
                stats["games"] += 1
                stats["wins"] += traj["win"]
                ot = traj["opp_type"]
                if ot in stats["by_opp"]:
                    stats["by_opp"][ot][0] += traj["win"]
                    stats["by_opp"][ot][1] += 1

                tag = "[green]WIN [/]" if traj["win"] else "[red]LOSS[/]"
                match_log.insert(
                    0,
                    f"{tag} vs {traj['opp_type']:8s} | "
                    f"HP {traj['my_hp']:2d}-{traj['opp_hp']:2d} | "
                    f"{traj['turns']} turns",
                )
                if len(match_log) > 12:
                    match_log.pop()

                buf_steps = sum(len(t["rewards"]) for t in buffer)
                if buf_steps < BATCH_SIZE:
                    continue

                # ---- PPO update ----
                stats["updates"] += 1
                phase_name, _ = get_phase(stats["updates"])
                stats["phase"] = phase_name

                # Adaptive LR and entropy coef
                current_lr  = get_lr(stats["updates"])
                current_ent = get_entropy_coef(stats["updates"])
                optimizer.learning_rate = current_lr

                all_obs  = np.concatenate([t["states"]    for t in buffer])
                all_acts = np.concatenate([t["actions"]   for t in buffer])
                all_lp   = np.concatenate([t["log_probs"] for t in buffer])
                all_mask = np.concatenate([t["masks"]     for t in buffer])

                all_adv_list, all_ret_list = [], []
                for t in buffer:
                    adv, ret = compute_gae(
                        t["rewards"], t["values"], t["dones"], t["last_val"]
                    )
                    all_adv_list.append(adv)
                    all_ret_list.append(ret)
                all_adv = np.concatenate(all_adv_list)
                all_ret = np.concatenate(all_ret_list)
                all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

                ent_coef_mx = mx.array(current_ent)
                total_loss = 0.0
                n_mini = 0
                indices = np.arange(len(all_obs))

                for _ in range(PPO_EPOCHS):
                    np.random.shuffle(indices)
                    for start in range(0, len(all_obs), MINI_BATCH):
                        idx = indices[start: start + MINI_BATCH]
                        loss, grads = loss_and_grad(
                            master,
                            mx.array(all_obs[idx]),
                            mx.array(all_acts[idx]),
                            mx.array(all_lp[idx]),
                            mx.array(all_ret[idx]),
                            mx.array(all_adv[idx]),
                            mx.array(all_mask[idx]),
                            ent_coef_mx,
                        )
                        optimizer.update(master, grads)
                        mx.eval(master.parameters(), optimizer.state)
                        total_loss += float(loss.item())
                        n_mini += 1

                stats["loss"] = total_loss / max(1, n_mini)
                stats["lr"]   = current_lr
                stats["ent"]  = current_ent

                # Checkpoint + league
                if stats["updates"] % CHECKPOINT_EVERY == 0:
                    cp = CHECKPOINT_DIR / f"step_{stats['updates']:07d}.npz"
                    mx.savez(
                        str(cp),
                        **dict(nn.utils.tree_flatten(master.parameters())),
                    )
                    wr = stats["wins"] / max(1, stats["games"])
                    league_pool.append((cp, stats["updates"], wr))
                    if len(league_pool) > LEAGUE_POOL_SIZE:
                        league_pool.pop(0)

                broadcast(stats["updates"])
                buffer = []

                # ---- UI ----
                wr = stats["wins"] / max(1, stats["games"])
                layout["header"].update(Panel(
                    f"[bold cyan]TITAN V4  |  Phase: {stats['phase']}  |  "
                    f"Updates: {stats['updates']}  |  Games: {stats['games']}  |  "
                    f"WR: {wr:.1%}  |  Loss: {stats['loss']:.4f}  |  "
                    f"LR: {stats['lr']:.1e}  |  Ent: {stats['ent']:.3f}[/]"
                ))

                m = Table(expand=True)
                m.add_column("Opponent"); m.add_column("Wins"); m.add_column("Games"); m.add_column("WR")
                for opp, (w2, g2) in stats["by_opp"].items():
                    m.add_row(opp, str(w2), str(g2), f"{w2/max(1,g2):.1%}")
                layout["metrics"].update(Panel(m, title="Win-rates by opponent type"))

                lt = Table(expand=True, show_header=False)
                for entry in match_log:
                    lt.add_row(entry)
                layout["log"].update(Panel(lt, title="Recent matches"))

            except KeyboardInterrupt:
                break
            except Exception:
                console.print_exception(show_locals=False)

    cancel_event.set()
    for w in workers:
        w.join(timeout=5)

    final = CHECKPOINT_DIR / "final.npz"
    mx.savez(str(final), **dict(nn.utils.tree_flatten(master.parameters())))
    console.print(f"[bold green]Done. Final model saved → {final}[/]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="TITAN V4 Training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .npz checkpoint to resume from")
    args = parser.parse_args()
    train(resume=args.resume)
