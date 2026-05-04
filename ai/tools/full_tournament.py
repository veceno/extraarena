import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Настройка путей
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as nn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ai.arena_env import ArenaEnv

console = Console()

# --- МОДЕЛЬ И АГЕНТ ---

class UniversalPolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, use_fc3: bool):
        super().__init__()
        self.use_fc3 = use_fc3
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.use_fc3:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        if self.use_fc3:
            x = nn.relu(self.fc3(x))
        return self.policy_head(x)

class SmartAgent:
    def __init__(self, path: str):
        weights = mx.load(path)
        h_dim, o_dim = weights["fc1.weight"].shape
        use_fc3 = "fc3.weight" in weights
        self.obs_dim = o_dim
        self.model = UniversalPolicyNetwork(o_dim, 256, h_dim, use_fc3)
        self.model.load_weights(path)
        mx.eval(self.model.parameters())

    def get_action(self, obs, mask):
        # Адаптация под размер входа модели (621 vs 997)
        if len(obs) > self.obs_dim: obs = obs[:self.obs_dim]
        elif len(obs) < self.obs_dim: obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        
        obs_mx = mx.array(obs)[None]
        logits = self.model(obs_mx)
        mask_mx = mx.array(mask)[None]
        logits = mx.where(mask_mx > 0, logits, -1e9)
        return int(mx.argmax(logits, axis=-1)[0])

# --- ЛОГИКА ЧЕСТНОГО МАТЧА (MIRROR) ---

def worker_play_match(p1_path: str, p2_path: str, p1_name: str, p2_name: str, match_seed: int):
    """Запускает две игры на одном сиде, меняя игроков местами."""
    mx.set_default_device(mx.cpu) 
    
    agent_a = SmartAgent(p1_path)
    agent_b = SmartAgent(p2_path)
    env = ArenaEnv(use_mlx=True)
    
    match_results = []
    
    # Раунд 1: A (P1) vs B (P2)
    obs, info = env.reset(seed=match_seed)
    done = False
    while not done:
        mask = info["legal_action_mask"]
        curr_id = env._engine.state.current_turn_owner_id
        agent = agent_a if curr_id == 1 else agent_b
        action = agent.get_action(obs, mask)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    match_results.append((p1_name if env._engine.state.status.name == "P1_WIN" else p2_name, p1_name, p2_name))
    
    # Раунд 2: B (P1) vs A (P2) на ТОМ ЖЕ СИДЕ
    obs, info = env.reset(seed=match_seed)
    done = False
    while not done:
        mask = info["legal_action_mask"]
        curr_id = env._engine.state.current_turn_owner_id
        # Меняем роли: теперь агент_b за P1
        agent = agent_b if curr_id == 1 else agent_a
        action = agent.get_action(obs, mask)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    match_results.append((p2_name if env._engine.state.status.name == "P1_WIN" else p1_name, p1_name, p2_name))
    
    return match_results

# --- ТУРНИР ---

def run_parallel_tournament(checkpoint_path: str, games_per_pair: int = 100):
    checkpoints = sorted(list(Path(checkpoint_path).glob("*.npz")))
    cp_data = [(str(cp), cp.name) for cp in checkpoints]
    
    stats = {cp.name: {"elo": 1200, "wins": 0, "total": 0} for cp in checkpoints}
    
    pairs = []
    for i in range(len(cp_data)):
        for j in range(i + 1, len(cp_data)):
            pairs.append((cp_data[i], cp_data[j]))

    # Каждая пара играет games_per_pair раз, но каждая "партия" в воркере — это 2 игры (зеркало)
    num_sessions = games_per_pair // 2
    total_matches = len(pairs) * num_sessions * 2 
    
    console.print(f"[bold magenta]⚖️ Запуск Честного Зеркального Турнира на M4 Pro![/]")
    console.print(f"Пар: {len(pairs)} | Всего игр: {total_matches}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("Игры (Mirror Match)...", total=total_matches)
        
        with ProcessPoolExecutor() as executor:
            futures = []
            for p1, p2 in pairs:
                for _ in range(num_sessions):
                    m_seed = random.randint(0, 10**8)
                    futures.append(executor.submit(worker_play_match, p1[0], p2[0], p1[1], p2[1], m_seed))
            
            for future in as_completed(futures):
                mirror_results = future.result() # Список из 2-х результатов
                
                for winner, n1, n2 in mirror_results:
                    # Обновление Эло
                    r1, r2 = stats[n1]["elo"], stats[n2]["elo"]
                    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                    e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
                    
                    s1 = 1 if winner == n1 else 0
                    s2 = 1 - s1
                    
                    stats[n1]["elo"] += 32 * (s1 - e1)
                    stats[n2]["elo"] += 32 * (s2 - e2)
                    stats[n1]["wins"] += s1
                    stats[n2]["wins"] += s2
                    stats[n1]["total"] += 1
                    stats[n2]["total"] += 1
                    
                    progress.update(task, advance=1)

    # Итоговая таблица
    table = Table(title="🏆 РЕЙТИНГ ELO (FAIR MIRROR EDITION)")
    table.add_column("Модель", style="cyan")
    table.add_column("Elo", justify="center", style="bold green")
    table.add_column("Winrate", justify="right")
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['elo'], reverse=True)
    for name, s in sorted_stats:
        wr = (s['wins'] / s['total']) * 100 if s['total'] > 0 else 0
        table.add_row(name, f"{int(s['elo'])}", f"{wr:.1f}%")
    
    console.print(table)

if __name__ == "__main__":
    path = project_root / "checkpoints"
    # На M4 Pro можно смело ставить 300 (это даст 300 игр на пару в зеркальном режиме)
    run_parallel_tournament(str(path), games_per_pair=300)