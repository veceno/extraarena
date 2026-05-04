import os
import mlx.nn as nn
import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

# Импортируем архитектуру из твоего основного файла
from arena_env import ArenaEnv

console = Console()

# ============================================================================
# КОНФИГУРАЦИЯ БИТВЫ
# ============================================================================
EVAL_CONFIG = {
    "CURRENT": "models/mlx/midoriya_v3_ep20000000.npz", 
    "BASELINE": "models/mlx/aggro_midoriya_v1.npz", # Или старая классика
    "NUM_GAMES": 10000,
    "MAX_STEPS": 256
}

class FlexibleMLXNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, h_dim: int, use_fc3: bool):
        super().__init__()
        self.use_fc3 = use_fc3
        self.fc1 = nn.Linear(obs_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        if self.use_fc3:
            self.fc3 = nn.Linear(h_dim, h_dim)
        self.policy_head = nn.Linear(h_dim, action_dim)
        self.value_head = nn.Linear(h_dim, 1)
    
    def __call__(self, x):
        x = nn.gelu(self.fc1(x))
        x = nn.gelu(self.fc2(x))
        if self.use_fc3:
            x = nn.gelu(self.fc3(x))
        return self.policy_head(x), self.value_head(x)

def load_model_dynamic(path):
    weights = mx.load(path)
    # Авто-определение параметров
    is_v3 = "fc3.weight" in weights
    h_dim = weights["fc1.weight"].shape[0]
    input_dim = weights["fc1.weight"].shape[1]
    
    console.print(f"[dim]• Конфигурация {os.path.basename(path)}: {input_dim}in, {h_dim}h, {'3' if is_v3 else '2'} layers[/]")
    
    # Создаем модель с нужным флагом слоев
    model = FlexibleMLXNet(input_dim, 256, h_dim, use_fc3=is_v3)
    model.load_weights(path)
    return model, input_dim

def run_battle(config):
    console.print(Panel("[bold magenta]⚔️ EXTRA ARENA: BATTLE OF GENERATIONS ⚔️[/]", border_style="magenta"))
    
    env = ArenaEnv(use_mlx=False)
    mx.set_default_device(mx.gpu) # На M4 Pro только GPU

    # Динамическая загрузка
    p_curr, dim_curr = load_model_dynamic(config["CURRENT"])
    p_base, dim_base = load_model_dynamic(config["BASELINE"])
    
    stats = {"wins": 0, "losses": 0, "steps": []}

    # Цикл битвы с прогресс-баром
    for i in track(range(config["NUM_GAMES"]), description="Сражения в разгаре..."):
        curr_is_p1 = (i % 2 == 0)
        obs, info = env.reset()
        done, steps = False, 0
        
        while not done and steps < config["MAX_STEPS"]:
            mask = info["legal_action_mask"]
            turn = env._engine.state.current_turn_owner_id
            
            # Выбор активной модели и подрезка обсервации если нужно
            if (curr_is_p1 and turn == 1) or (not curr_is_p1 and turn == 2):
                active_net, dim = p_curr, dim_curr
            else:
                active_net, dim = p_base, dim_base
            
            # Срез обсервации под размер модели
            input_obs = obs[:dim] if obs.shape[0] > dim else obs
            
            logits, _ = active_net(mx.array(input_obs[None]))
            logits = mx.where(mx.array(mask[None]) > 0, logits, -1e9)
            action = int(mx.argmax(logits, axis=-1)[0]) # Greedy play
            
            obs, reward, term, trunc, info = env.step(action)
            done, steps = term or trunc, steps + 1
            
        # Результат
        winner = 1 if curr_is_p1 else 2
        if env._engine.state.status.name == f'P{winner}_WIN':
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["steps"].append(steps)

    # Красивая итоговая таблица
    winrate = stats["wins"] / config["NUM_GAMES"]
    table = Table(title="📊 ИТОГИ СЕРИИ", border_style="green")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="bold white")
    
    table.add_row("Текущая модель", os.path.basename(config["CURRENT"]))
    table.add_row("Оппонент", os.path.basename(config["BASELINE"]))
    table.add_row("Winrate", f"[bold green]{winrate:.1%}[/]")
    table.add_row("Avg Steps", f"{np.mean(stats['steps']):.1f}")
    
    console.print(table)

if __name__ == "__main__":
    run_battle(EVAL_CONFIG)