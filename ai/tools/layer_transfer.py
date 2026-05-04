import numpy as np
import mlx.core as mx
import os

def evolve_aggro():
    old_path = "checkpoints/aggro_midoriya_v1.npz"
    new_path = "checkpoints/aggro_v3_ready.npz"
    
    if not os.path.exists(old_path):
        print(f"❌ Файл {old_path} не найден!")
        return

    old = mx.load(old_path)
    new_weights = {}

    # 1. FC1: (621) -> (512, 997)
    # Копируем старые знания в начало, остальное заполняем легким шумом
    w1 = mx.random.normal((512, 997)) * 0.01
    w1_old = old["fc1.weight"]
    w1[:w1_old.shape[0], :w1_old.shape[1]] = w1_old
    new_weights["fc1.weight"] = w1
    
    b1 = mx.zeros((512,))
    b1[:old["fc1.bias"].shape[0]] = old["fc1.bias"]
    new_weights["fc1.bias"] = b1

    # 2. FC2: (512, 512) -> (512, 512)
    new_weights["fc2.weight"] = old["fc2.weight"]
    new_weights["fc2.bias"] = old["fc2.bias"]

    # 3. FC3: НОВЫЙ СЛОЙ (512, 512)
    # Делаем его почти "прозрачным" (Identity), чтобы не сломать логику V1 сразу
    new_weights["fc3.weight"] = mx.eye(512) * 0.9 + mx.random.normal((512, 512)) * 0.01
    new_weights["fc3.bias"] = mx.zeros((512,))

    # 4. Policy Head
    new_weights["policy_head.weight"] = old["policy_head.weight"]
    new_weights["policy_head.bias"] = old["policy_head.bias"]

    # 5. Value Head
    new_weights["value_head.weight"] = old["value_head.weight"]
    new_weights["value_head.bias"] = old["value_head.bias"]

    mx.savez(new_path, **new_weights)
    print(f"✅ Эволюция завершена! Создан сид: {new_path}")

if __name__ == "__main__":
    evolve_aggro()