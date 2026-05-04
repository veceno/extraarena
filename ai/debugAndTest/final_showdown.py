import os
import numpy as np
import mlx.core as mx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from pathlib import Path

# Импортируем твои классы
from evaluate_agents import SmartAgent
from arena_env import ArenaEnv

console = Console()

def run_final_battle(path_old, path_new, num_games=100):
    env = ArenaEnv(use_mlx=False)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    console.print(Panel.fit(
        "[bold marathon]🥊 ВЕЛИКАЯ БИТВА: 9M (TURTLE) vs 9.5M (BERSERK)[/]",
        border_style="magenta",
        subtitle="ExtraArena Championship"
    ))

    # Загружаем бойцов
    with console.status("[bold yellow]Подготовка бойцов на арене...") as status:
        agent_old = SmartAgent("9.0M_Turtle", path_old, obs_dim, action_dim)
        agent_new = SmartAgent("9.5M_Berserk", path_new, obs_dim, action_dim)

    stats = {
        agent_old.name: {"wins": 0, "hp": [], "turns": []},
        agent_new.name: {"wins": 0, "hp": [], "turns": []}
    }

    # Прогресс-бар матчей
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        battle_task = progress.add_task("[cyan]Проведение матчей...", total=num_games)

        for g in range(num_games):
            # Чередуем очередность ходов
            p1, p2 = (agent_old, agent_new) if g % 2 == 0 else (agent_new, agent_old)
            
            obs, info = env.reset(seed=777 + g)
            done = False
            m_turns = 0
            
            while not done and m_turns < 300:
                curr_p = p1 if env._engine.state.current_turn_owner_id == 1 else p2
                act = curr_p.get_action(obs, info["legal_action_mask"])
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
                m_turns += 1
            
            status_game = env._engine.state.status.name
            winner_name = ""
            if status_game == "P1_WIN":
                winner_name, win_hp = p1.name, env._engine.state.p1.hero.hp
            elif status_game == "P2_WIN":
                winner_name, win_hp = p2.name, env._engine.state.p2.hero.hp

            if winner_name:
                stats[winner_name]["wins"] += 1
                stats[winner_name]["hp"].append(win_hp)
                stats[winner_name]["turns"].append(m_turns)

            progress.update(battle_task, advance=1)

    # Итоговая таблица
    table = Table(title="📊 ИТОГИ ТУРНИРА", show_lines=True, header_style="bold cyan")
    table.add_column("Модель", style="bold")
    table.add_column("Побед", justify="center", style="green")
    table.add_column("Win Rate", justify="center")
    table.add_column("Ср. HP", justify="right", style="yellow")
    table.add_column("Ср. ходов", justify="right", style="magenta")

    for name, s in stats.items():
        wr = (s["wins"] / num_games) * 100
        avg_hp = np.mean(s["hp"]) if s["hp"] else 0
        avg_turns = np.mean(s["turns"]) if s["turns"] else 0
        table.add_row(
            name,
            str(s["wins"]),
            f"{wr:.1f}%",
            f"{avg_hp:.1f}",
            f"{avg_turns:.1f}"
        )

    console.print(table)

    # Финальный вердикт
    winner_overall = max(stats, key=lambda x: stats[x]["wins"])
    win_diff = abs(stats[agent_old.name]["wins"] - stats[agent_new.name]["wins"])
    
    verdict_style = "bold green" if "Berserk" in winner_overall else "bold yellow"
    console.print(Panel(
        f"🏆 Победитель: [{verdict_style}]{winner_overall}[/]\n"
        f"Разрыв: {win_diff} побед(ы)\n"
        f"Статус: {'🔥 Агрессия победила!' if 'Berserk' in winner_overall else '🛡️ Осторожность взяла верх!'}",
        title="ИТОГОВЫЙ ВЕРДИКТ",
        border_style=verdict_style
    ))

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    # Путь к "Золотой" старой модели
    p_old = str(root / "checkpoints_selfplay" / "GOLD_MIDORIYA_9M.npz")
    # Путь к новой агрессивной модели (из твоего последнего лога)
    p_new = str(root / "checkpoints_selfplay" / "policy_selfplay_ep500016.npz")
    
    run_final_battle(p_old, p_new, num_games=100)