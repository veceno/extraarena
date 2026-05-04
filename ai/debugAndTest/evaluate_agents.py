import os
import sys
import glob
import re
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

# Импорты Rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.live import Live

# Настройка путей
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ai.arena_env import ArenaEnv

console = Console()

class MLXPolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Добавляем fc3, чтобы соответствовать весам в .npz файлах
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) # Тот самый недостающий слой
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def __call__(self, x):
        x = mx.maximum(self.fc1(x), 0)
        x = mx.maximum(self.fc2(x), 0)
        x = mx.maximum(self.fc3(x), 0) # Прогоняем данные через него тоже
        return self.policy_head(x)

class SmartAgent:
    def __init__(self, name: str, path: str = None, obs_dim=None, action_dim=None):
        self.name = name
        self.policy = None
        self.weight_hash = 0.0
        if path:
            self.policy = MLXPolicyNetwork(obs_dim, action_dim)
            weights = mx.load(path)
            tree = nn.utils.tree_unflatten(list(weights.items()))
            self.policy.update(tree)
            mx.eval(self.policy.parameters())
            
            flat_params = nn.utils.tree_flatten(self.policy.parameters())
            self.weight_hash = sum([mx.sum(p).item() for _, p in flat_params])

    def get_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if not self.policy:
            actions = np.where(mask > 0)[0]
            return np.random.choice(actions) if len(actions) > 0 else 0
        
        obs_mlx = mx.array(obs.reshape(1, -1))
        logits = self.policy(obs_mlx)
        mask_mlx = mx.array(mask.reshape(1, -1))
        logits = mx.where(mask_mlx > 0, logits, -1e9)
        return int(mx.argmax(logits[0]))

def run_mega_tournament(checkpoint_dir: str, num_games: int = 100):
    env = ArenaEnv(use_mlx=False)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    console.print(Panel.fit("[bold cyan]🏟️ EXTRAARENA MEGA TOURNAMENT[/bold cyan]", border_style="cyan"))

    # Сбор чекпоинтов
    all_files = glob.glob(os.path.join(checkpoint_dir, "*.npz"))
    def get_ep(f):
        nums = re.findall(r'\d+', os.path.basename(f))
        return int(nums[-1]) if nums else 0
    all_files.sort(key=get_ep)
    
    # Отбираем 9M и несколько промежуточных для сравнения
    if len(all_files) > 6:
        indices = np.linspace(0, len(all_files)-1, 6, dtype=int)
        selected_files = [all_files[i] for i in indices]
    else:
        selected_files = all_files

    agents = [SmartAgent("Random")]
    
    with console.status("[bold green]Загрузка агентов в память M4 Pro...") as status:
        for f in selected_files:
            ep_num = get_ep(f)
            agent = SmartAgent(f"EP_{ep_num}", f, obs_dim, action_dim)
            agents.append(agent)
            console.log(f"✅ [bold]{agent.name}[/] загружен (Hash: {agent.weight_hash:.2f})")

    stats = {a.name: {"wins": 0, "elo": 1200, "hp": [], "turns": []} for a in agents}
    pairs = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            pairs.append((agents[i], agents[j]))

    # Прогресс-бар турнира
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        overall_task = progress.add_task("[yellow]Общий прогресс турнира", total=len(pairs) * num_games)

        for a1, a2 in pairs:
            pair_seed_base = (agents.index(a1) * 777) + (agents.index(a2) * 123)
            a1_wins = 0
            
            for g in range(num_games):
                p1, p2 = (a1, a2) if g % 2 == 0 else (a2, a1)
                obs, info = env.reset(seed=pair_seed_base + g)
                done = False
                m_turns = 0
                
                while not done and m_turns < 300:
                    curr_p = p1 if env._engine.state.current_turn_owner_id == 1 else p2
                    act = curr_p.get_action(obs, info["legal_action_mask"])
                    obs, _, term, trunc, info = env.step(act)
                    done = term or trunc
                    m_turns += 1
                
                status_game = env._engine.state.status.name
                winner = None
                if status_game == "P1_WIN": winner, win_hp = p1, env._engine.state.p1.hero.hp
                elif status_game == "P2_WIN": winner, win_hp = p2, env._engine.state.p2.hero.hp

                if winner:
                    stats[winner.name]["wins"] += 1
                    stats[winner.name]["hp"].append(win_hp)
                    stats[winner.name]["turns"].append(m_turns)
                    if winner.name == a1.name: a1_wins += 1
                
                progress.update(overall_task, advance=1, description=f"⚔️ [bold]{a1.name}[/] vs [bold]{a2.name}[/]")

            # Обновление Эло
            wr_a1 = a1_wins / num_games
            exp_a1 = 1 / (1 + 10 ** ((stats[a2.name]["elo"] - stats[a1.name]["elo"]) / 400))
            stats[a1.name]["elo"] += 100 * (wr_a1 - exp_a1)
            stats[a2.name]["elo"] += 100 * ((1 - wr_a1) - (1 - exp_a1))

    # Финальная таблица
    table = Table(title="🏆 ИТОГОВЫЙ РЕЙТИНГ (9M EDITION)", title_style="bold magenta", show_lines=True)
    table.add_column("Агент", style="cyan", no_wrap=True)
    table.add_column("Эло", justify="center", style="bold green")
    table.add_column("Побед", justify="right", style="yellow")
    table.add_column("Ср. HP", justify="right")
    table.add_column("Ср. ходов", justify="right")

    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['elo'], reverse=True)
    for name, s in sorted_stats:
        avg_hp = np.mean(s['hp']) if s['hp'] else 0
        avg_turns = np.mean(s['turns']) if s['turns'] else 0
        table.add_row(
            name, 
            str(int(s['elo'])), 
            str(s['wins']), 
            f"{avg_hp:.1f}", 
            f"{avg_turns:.1f}"
        )

    console.print(table)

if __name__ == "__main__":
    # Укажи путь к твоим 9M чекпоинтам
    path = str(project_root / "checkpoints_selfplay")
    run_mega_tournament(path, num_games=100)