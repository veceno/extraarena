"""
ai/debug_match.py
Быстрая проверка логики игры без обучения.
"""
import sys
import os
import time
import numpy as np

# Добавляем путь к корню
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.arena_env import ArenaEnv

def run_debug():
    print("⚡ Init ArenaEnv...")
    try:
        env = ArenaEnv()
        obs, info = env.reset()
        print("✅ Environment Created!")
    except Exception as e:
        print(f"❌ Error creating env: {e}")
        return

    print("\n⚔️  STARTING MATCH ⚔️")
    done = False
    step = 0
    
    while not done:
        step += 1
        
        # Получаем маску легальных ходов
        mask = env.get_action_mask()
        valid_actions = np.where(mask > 0)[0]
        
        if len(valid_actions) == 0:
            print("💀 No valid actions (BUG!)")
            break
            
        # Случайный ход
        action = np.random.choice(valid_actions)
        
        # Декодируем для красоты (если бы был метод decode, но мы просто принтим ID)
        action_type = "END TURN" if action == 0 else ("PLAY" if action <= 170 else "ATTACK")
        
        print(f"Turn {env.engine.state.turn_number} | Step {step} | Player {env.engine.state.current_turn_owner_id} performs action {action} ({action_type})")
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            print(f"\n🏆 Game Over! Winner: {env.engine.state.status}")
            print(f"Total Steps: {step}")

if __name__ == "__main__":
    run_debug()