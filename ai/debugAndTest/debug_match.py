import numpy as np
import mlx.core as mx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from evaluate_agents import SmartAgent, MLXPolicyNetwork
from arena_env import ArenaEnv
from pathlib import Path

console = Console()

def detailed_debug_duel(path_turtle, path_berserk):
    env = ArenaEnv(use_mlx=False)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Загружаем наших финалистов
    turtle = SmartAgent("9.0M_Turtle", path_turtle, obs_dim, action_dim)
    berserk = SmartAgent("9.5M_Berserk", path_berserk, obs_dim, action_dim)

    # Используем фиксированный сид для воспроизводимости или случайный для новых тестов
    seed = np.random.randint(0, 10000)
    obs, info = env.reset(seed=seed)
    done = False
    turn = 0
    
    last_p1_hp = env._engine.state.p1.hero.hp
    last_p2_hp = env._engine.state.p2.hero.hp

    table = Table(title=f"⚔️ BATTLE LOG: Turtle vs Berserk (Seed: {seed})", show_header=True, header_style="bold magenta")
    table.add_column("Ход", justify="right", style="dim")
    table.add_column("Игрок", justify="center")
    table.add_column("Действие", style="cyan")
    table.add_column("Уверенность", justify="right")
    table.add_column("Урон", justify="center", style="bold yellow")
    table.add_column("HP (T | B)", justify="center")

    while not done and turn < 150:
        # В этом тесте P1 - Turtle, P2 - Berserk
        is_turtle = env._engine.state.current_turn_owner_id == 1
        curr_agent = turtle if is_turtle else berserk
        
        mask = info["legal_action_mask"]
        obs_mlx = mx.array(obs.reshape(1, -1))
        logits = curr_agent.policy(obs_mlx)
        probs = np.array(mx.softmax(logits))[0]
        
        action = curr_agent.get_action(obs, mask)
        confidence = probs[action]

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
        p1_hp = env._engine.state.p1.hero.hp
        p2_hp = env._engine.state.p2.hero.hp
        
        # Считаем нанесенный урон
        d1 = last_p1_hp - p1_hp
        d2 = last_p2_hp - p2_hp
        dmg_str = f"T:-{d1}" if d1 > 0 else (f"B:-{d2}" if d2 > 0 else "---")
        
        # Подсветка агрессивных ходов Берсерка
        act_style = "bold green" if not is_turtle and action != 0 else "white"
        
        table.add_row(
            str(turn),
            "[bold blue]Turtle[/]" if is_turtle else "[bold red]Berserk[/]",
            f"[{act_style}]ID:{action}[/]",
            f"{confidence:.1%}",
            dmg_str,
            f"{p1_hp} | {p2_hp}"
        )
        
        last_p1_hp, last_p2_hp = p1_hp, p2_hp
        turn += 1

    console.print(table)
    console.print(Panel(f"[bold yellow]РЕЗУЛЬТАТ:[/] {env._engine.state.status.name}\n[bold]Всего ходов:[/] {turn}", border_style="yellow"))

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    # Пути к твоим актуальным моделям
    p_turtle = str(root / "checkpoints_selfplay" / "GOLD_MIDORIYA_9M.npz")
    p_berserk = str(root / "checkpoints_selfplay" / "policy_selfplay_ep500016.npz")
    detailed_debug_duel(p_turtle, p_berserk)