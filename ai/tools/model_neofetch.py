import mlx.core as mx
import platform
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.align import Align

console = Console()

# МАКСИМАЛЬНО ДЕТАЛИЗИРОВАННЫЙ АРТ "BERSERK-X"
# Используем спецсимволы и цветовые акценты для "злого" взгляда
ASCII_ART = r"""
[bold red]              .        .              [/]
[bold red]             / \      / \             [/]
[bold red]            /   \____/   \            [/]
[bold red]           /  [bold white]__[/][bold red]      [bold white]__[/][bold red]  \           [/]
[bold red]          /  [bold yellow]/  \[/][bold red]    [bold yellow]/  \[/][bold red]  \          [/]
[bold red]         |  [bold yellow]| () |[/][bold red]  [bold yellow]| () |[/][bold red]  |         [/]
[bold red]         |   [bold yellow]\__/    \__/[/][bold red]   |         [/]
[bold red]         |       [bold white]____[/][bold red]       |         [/]
[bold red]          \     [bold white][____][/][bold red]     /          [/]
[bold red]           \ [bold white]__________[/][bold red] /           [/]
[bold red]         [bold black]___[/][bold red]/            \[[bold black]___[/]          [/]
[bold red]      [bold black]-=[[/][bold white]  MIDORIYA  V2.0  [/][bold black]]=-[/]        [/]
[bold red]         [bold black]---[/][bold red]\__________/[bold black]---[/]          [/]
[bold red]               ||    ||               [/]
"""

def get_model_info():
    # Генерируем статы на основе твоих прогонов
    info = [
        ("PROJECT", "ExtraArena: Evolution"),
        ("CODENAME", "Midoriya Absolute"),
        ("STATUS", "[bold green]EVOLVED & OPTIMIZED[/]"),
        ("ELO PEAK", "1621 [bold green]▲ TOP 1[/]"),
        ("NET TYPE", "Triple-Layer Perceptron"),
        ("WIDTH", "512 Neurons per Layer"),
        ("SENSES", "997 Input Features"),
        ("TRAINED", "9.0M + Evolved V2"),
        ("COMPUTE", f"{platform.processor()} (M4 Pro)"),
        ("MEMORY", f"{round(psutil.virtual_memory().total / (1024**3), 1)} GB"),
        ("KERNEL", platform.system()),
    ]
    
    text = Text()
    for key, val in info:
        # Добавляем стильные точки-разделители
        text.append(f"{key:.<18}", style="bright_black")
        text.append(f" {val}\n", style="white")
    return text

# Формируем и выводим панель
console.print("\n")
console.print(Panel(
    Columns([
        Align.center(ASCII_ART, vertical="middle"), 
        get_model_info()
    ], padding=(0, 4)),
    title="[bold red] ⚡ NEURAL CORE IDENTIFIED ⚡ [/]",
    subtitle="[bold blue] ACCESS GRANTED: BERSERK PROTOCOL [/]",
    border_style="bright_blue",
    expand=False,
    padding=(1, 2)
))