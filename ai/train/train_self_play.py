"""
TITAN V3: ROBUST TRAINING (ANTI-CRASH)
--------------------------------------
- Ignored queue timeouts (will wait forever if needed).
- Added worker status logging.
- Fixed crashes on empty queue.
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import traceback

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from ai.arena_env import ArenaEnv
    from core.state import GameStatus
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ai.arena_env import ArenaEnv
    from core.state import GameStatus

# --- CONFIG ---
HIDDEN_DIM = 512
NUM_LAYERS = 5
LEARNING_RATE = 3e-4
NUM_ACTORS = 8          # Снизим до 8 для стабильности на старте
STEPS_PER_ACTOR = 256
BATCH_SIZE = NUM_ACTORS * STEPS_PER_ACTOR
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# --- MODEL ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
    def __call__(self, x):
        return self.act(self.norm(self.fc(x))) + x

class TitanNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, HIDDEN_DIM)
        self.blocks = [ResBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)]
        self.actor_head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, act_dim))
        self.critic_head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 1))
    def __call__(self, x):
        x = nn.SiLU()(self.input_layer(x))
        for block in self.blocks: x = block(x)
        return self.actor_head(x), self.critic_head(x)

# --- WORKER ---
def worker_fn(worker_id, weights_queue, traj_queue, cancel_event):
    print(f"🔧 Worker {worker_id} started...")
    try:
        env = ArenaEnv()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        local_net = TitanNet(obs_dim, act_dim)
        opp_net = TitanNet(obs_dim, act_dim)
        
        while not cancel_event.is_set():
            # Sync Weights
            try:
                if not weights_queue.empty():
                    params, stage = weights_queue.get()
                    loaded = {k: mx.array(v) for k, v in params.items()}
                    local_net.load_weights(list(loaded.items()))
                    if stage == 1 or (stage == 3 and random.random() < 0.5):
                         opp_net.load_weights(list(loaded.items()))
            except Exception: pass

            obs, _ = env.reset()
            states, actions, rewards, values, log_probs, masks = [], [], [], [], [], []
            done = False
            step = 0
            
            while not done and step < STEPS_PER_ACTOR:
                obs_mx = mx.array(obs.reshape(1, -1))
                mask = env.get_action_mask()
                logits, val = local_net(obs_mx)
                logits = logits + (1.0 - mx.array(mask)) * -1e9
                probs = np.array(nn.softmax(logits))[0]
                probs /= (probs.sum() + 1e-8)
                
                action = np.random.choice(len(probs), p=probs)
                
                if env.engine.state.current_turn_owner_id == env.agent_id:
                    log_prob = np.log(probs[action] + 1e-10)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    states.append(obs); actions.append(action); rewards.append(reward)
                    values.append(val.item()); log_probs.append(log_prob); masks.append(mask)
                    obs = next_obs
                    done = terminated or truncated
                    step += 1
                else:
                    opp_mask = mx.array(env.get_action_mask())
                    opp_logits, _ = opp_net(obs_mx)
                    opp_logits = opp_logits + (1.0 - opp_mask) * -1e9
                    opp_action = mx.argmax(opp_logits, axis=1).item()
                    next_obs, _, terminated, truncated, _ = env.step(opp_action)
                    obs = next_obs
                    done = terminated or truncated
            
            # End of episode/batch
            last_val = 0
            if not done:
                _, val = local_net(mx.array(obs.reshape(1, -1)))
                last_val = val.item()
                
            win = 0
            st = env.engine.state
            if st.status == GameStatus.P1_WIN and env.agent_id == st.p1.user_id: win = 1
            elif st.status == GameStatus.P2_WIN and env.agent_id == st.p2.user_id: win = 1
                
            traj = {
                'states': np.array(states), 'actions': np.array(actions),
                'rewards': np.array(rewards), 'values': np.array(values),
                'log_probs': np.array(log_probs), 'masks': np.array(masks),
                'dones': done, 'last_val': last_val, 'win': win,
                'opp_tag': "SELF", 'turns': st.turn_number,
                'my_hp': st.p1.hero.hp if env.agent_id==st.p1.user_id else st.p2.hero.hp,
                'opp_hp': st.p2.hero.hp if env.agent_id==st.p1.user_id else st.p1.hero.hp
            }
            traj_queue.put(traj) # Block until slot is free
            
    except Exception as e:
        print(f"❌ Worker {worker_id} DIED:")
        traceback.print_exc()

def compute_gae(rewards, values, dones, next_value):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t] if t == len(rewards)-1 else 1.0
        nextvalues = next_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
    return advantages, advantages + values

def train_universal(args):
    if not CHECKPOINT_DIR.exists(): CHECKPOINT_DIR.mkdir(parents=True)
    
    # Init Env
    dummy_env = ArenaEnv()
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.n
    
    master_policy = TitanNet(obs_dim, act_dim)
    mx.eval(master_policy.parameters())
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    
    # Clean checkpoints
    for cp in CHECKPOINT_DIR.glob("*.npz"):
        try:
            w = mx.load(str(cp))
            if w['input_layer.weight'].shape[0] != obs_dim: os.remove(cp)
        except: pass

    weights_queue = mp.Queue()
    traj_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    cancel_event = mp.Event()
    
    print(f"🚀 Starting {NUM_ACTORS} workers...")
    workers = [mp.Process(target=worker_fn, args=(i, weights_queue, traj_queue, cancel_event)) for i in range(NUM_ACTORS)]
    for p in workers: p.start()
    
    # Initial weights
    init_w = {k: np.array(v) for k, v in dict(nn.utils.tree_flatten(master_policy.parameters())).items()}
    for _ in range(NUM_ACTORS): weights_queue.put((init_w, 0))
    
    console = Console()
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="metrics", size=10), Layout(name="log", ratio=1))
    
    stats = {'updates': 0, 'w_self': 0, 'g_self': 0, 'w_league': 0, 'g_league': 0, 'loss': 0}
    match_log = []
    
    with Live(layout, refresh_per_second=2, console=console):
        buffer = []
        while stats['updates'] < 100000000:
            try:
                # --- MAIN FIX: Handle Empty Queue without crashing ---
                try:
                    traj = traj_queue.get(timeout=2.0)
                except Empty:
                    # Если очередь пуста, просто обновляем UI и ждем дальше
                    layout["header"].update(Panel(f"[bold yellow]WAITING FOR DATA... ({len(buffer)}/{BATCH_SIZE} steps)[/]", style="yellow"))
                    continue
                
                buffer.append(traj)
                
                # Update UI Stats
                if traj['opp_tag'] == "SELF":
                    stats['g_self'] += 1; stats['w_self'] += traj['win']
                else:
                    stats['g_league'] += 1; stats['w_league'] += traj['win']
                
                outcome = "[green]WIN [/]" if traj['win'] else "[red]LOSS[/]"
                match_log.insert(0, f"{outcome} HP {traj['my_hp']}-{traj['opp_hp']} | {traj['turns']} trn")
                if len(match_log) > 8: match_log.pop()
                
                # Check Batch Size
                current_steps = sum(len(t['rewards']) for t in buffer)
                if current_steps >= BATCH_SIZE:
                    stats['updates'] += 1
                    layout["header"].update(Panel(f"[bold cyan]UPDATING NETWORK... ({stats['updates']})[/]", style="cyan"))
                    
                    # PPO Update Logic
                    all_obs = np.concatenate([x['states'] for x in buffer])
                    all_acts = np.concatenate([x['actions'] for x in buffer])
                    all_lp = np.concatenate([x['log_probs'] for x in buffer])
                    all_rew = np.concatenate([x['rewards'] for x in buffer])
                    all_val = np.concatenate([x['values'] for x in buffer])
                    all_mask = np.concatenate([x['masks'] for x in buffer])
                    
                    all_adv, all_ret = [], []
                    for t in buffer:
                        adv, ret = compute_gae(t['rewards'], t['values'], [t['dones']], t['last_val'])
                        all_adv.append(adv); all_ret.append(ret)
                    all_adv = np.concatenate(all_adv); all_ret = np.concatenate(all_ret)
                    all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
                    
                    def loss_fn(m, x, a, lp_old, ret, adv, mask):
                        l, v = m(x)
                        l = l + (1.0 - mask) * -1e9
                        probs = nn.softmax(l, axis=-1)
                        a_hot = mx.zeros_like(l)
                        a_hot[mx.arange(a.shape[0]), a] = 1.0
                        pi = mx.sum(probs * a_hot, axis=1)
                        ratio = mx.exp(mx.log(pi + 1e-10) - lp_old)
                        surr1 = ratio * adv
                        surr2 = mx.clip(ratio, 0.8, 1.2) * adv
                        return -mx.mean(mx.minimum(surr1, surr2)) + 0.5 * mx.mean((ret - v.squeeze())**2)

                    indices = np.arange(len(all_obs))
                    np.random.shuffle(indices)
                    for start in range(0, len(all_obs), 512):
                        idx = mx.array(indices[start:start+512])
                        l, g = nn.value_and_grad(master_policy, loss_fn)(
                            master_policy, mx.array(all_obs[idx]), mx.array(all_acts[idx]), 
                            mx.array(all_lp[idx]), mx.array(all_ret[idx]), 
                            mx.array(all_adv[idx]), mx.array(all_mask[idx])
                        )
                        optimizer.update(master_policy, g)
                        mx.eval(master_policy.parameters(), optimizer.state)
                    
                    stats['loss'] = float(l)
                    
                    # Broadcast
                    new_w = {k: np.array(v) for k, v in dict(nn.utils.tree_flatten(master_policy.parameters())).items()}
                    while not weights_queue.empty(): 
                        try: weights_queue.get_nowait()
                        except: break
                    for _ in range(NUM_ACTORS): weights_queue.put((new_w, 0))
                    
                    if stats['updates'] % 50 == 0:
                        mx.savez(str(CHECKPOINT_DIR / f"step_{stats['updates']}.npz"), **dict(nn.utils.tree_flatten(master_policy.parameters())))
                    
                    buffer = []
                
                # Render
                layout["header"].update(Panel(f"[bold cyan]TITAN V3 | Updates: {stats['updates']} | Buffer: {current_steps}/{BATCH_SIZE}[/]", style="cyan"))
                m_table = Table(expand=True); m_table.add_column("Metric"); m_table.add_column("Value")
                m_table.add_row("Winrate", f"{stats['w_self']/max(1, stats['g_self']):.1%}")
                m_table.add_row("Loss", f"{stats['loss']:.4f}")
                layout["metrics"].update(Panel(m_table))
                l_table = Table(expand=True, show_header=False)
                for e in match_log: l_table.add_row(e)
                layout["log"].update(Panel(l_table))
                    
            except KeyboardInterrupt: break
            except Exception:
                console.print_exception(show_locals=True)

    cancel_event.set()
    for p in workers: p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    train_universal(args)