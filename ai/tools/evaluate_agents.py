import sys
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# 1. ХАК ДЛЯ ПУТЕЙ: Добавляем корень проекта в sys.path ДО всех импортов
# Это позволяет находить модуль 'core' из папки 'ai/'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as nn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# 2. ТЕПЕРЬ ИМПОРТЫ СРАБОТАЮТ
from ai.arena_env import ArenaEnv
from core.engine import ArenaEnvironment

console = Console()

class UniversalPolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, use_fc3: bool):
        super().__init__()
        self.use_fc3 = use_fc3
        # Динамические размеры на основе входных данных
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
    def __init__(self, name: str, path: str, action_dim: int = 256):
        self.name = name
        # Загружаем веса, чтобы прочитать их форму
        weights = mx.load(path)
        
        # Авто-детекция архитектуры из весов первого слоя
        # В MLX форма fc1.weight это (hidden_dim, obs_dim)
        h_dim, o_dim = weights["fc1.weight"].shape
        use_fc3 = "fc3.weight" in weights
        
        self.obs_dim = o_dim
        self.model = UniversalPolicyNetwork(o_dim, action_dim, h_dim, use_fc3)
        self.model.load_weights(path)
        mx.eval(self.model.parameters())
        
        type_str = "Берсерк" if use_fc3 else "Старая"
        print(f"📦 {name}: {type_str} (Вход: {o_dim}, Слой: {h_dim})")

    def get_action(self, obs, mask):
        # Если текущее наблюдение (997) больше, чем модель умеет видеть (621),
        # мы просто обрезаем лишнее, чтобы старая модель не "ослепла"
        if len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
        elif len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))

        obs_mx = mx.array(obs)[None]
        logits = self.model(obs_mx)
        mask_mx = mx.array(mask)[None]
        logits = mx.where(mask_mx > 0, logits, -1e9)
        return int(mx.argmax(logits, axis=-1)[0])

def run_mega_tournament(checkpoint_path: str, num_games: int = 50):
    # Собираем все .npz файлы
    checkpoints = sorted(list(Path(checkpoint_path).glob("*.npz")))
    
    if not checkpoints:
        console.print(f"[bold red]Ошибка:[/] В папке {checkpoint_path} не найдено .npz файлов!")
        return

    # Инициализируем таблицу результатов
    results_table = Table(title=f"Турнир моделей ExtraArena (по {num_games} игр)")
    results_table.add_column("Модель", style="cyan", no_wrap=True)
    results_table.add_column("Winrate (%)", justify="right", style="green")
    results_table.add_column("Avg HP", justify="right", style="magenta")
    results_table.add_column("Avg Turns", justify="right", style="yellow")

    results = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Проведение матчей...", total=len(checkpoints))
        
        for cp in checkpoints:
            try:
                agent = SmartAgent(cp.name, str(cp))
                # Создаем окружение для тестов
                env = ArenaEnv(use_mlx=True)
                
                wins = 0
                total_hp = 0
                total_turns = 0
                
                for _ in range(num_games):
                    obs, info = env.reset()
                    done = False
                    while not done:
                        mask = info["legal_action_mask"]
                        action = agent.get_action(obs, mask)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                    
                    # Считаем статистику по окончании игры
                    agent_player = env._get_agent_player()
                    if agent_player.hero.hp > 0 and env._engine.state.status.name.endswith("WIN"):
                        wins += 1
                    total_hp += agent_player.hero.hp
                    total_turns += env._turn_count
                
                results.append({
                    "name": cp.name,
                    "winrate": (wins / num_games) * 100,
                    "avg_hp": total_hp / num_games,
                    "avg_turns": total_turns / num_games
                })
            except Exception as e:
                console.print(f"[red]Ошибка при тестировании {cp.name}: {e}[/]")
            
            progress.update(task, advance=1)

    # Сортировка по винрейту и вывод
    results.sort(key=lambda x: x["winrate"], reverse=True)
    for r in results:
        results_table.add_row(
            r["name"], 
            f"{r['winrate']:.1f}%", 
            f"{r['avg_hp']:.1f}", 
            f"{r['avg_turns']:.1f}"
        )
    
    console.print(results_table)

if __name__ == "__main__":
    # Указываем путь к папке checkpoints в корне проекта
    target_path = project_root / "checkpoints"
    
    console.print(f"[bold blue]Корень проекта:[/] {project_root}")
    console.print(f"[bold blue]Папка моделей:[/] {target_path}")
    
    if not target_path.exists():
        console.print(f"[bold red]Ошибка:[/] Папка {target_path} не существует!")
    else:
        run_mega_tournament(str(target_path), num_games=50)