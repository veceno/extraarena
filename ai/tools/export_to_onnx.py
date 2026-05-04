import torch
import torch.nn as nn
import numpy as np
import os

class UniversalTorchNet(nn.Module):
    def __init__(self, obs_dim, action_dim=256, hidden_dim=512, use_fc3=True):
        super().__init__()
        self.use_fc3 = use_fc3
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.use_fc3:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if self.use_fc3:
            x = torch.relu(self.fc3(x))
        return self.policy_head(x)

def export(checkpoint_path):
    # 1. Загружаем веса MLX
    weights = np.load(checkpoint_path)
    
    # Авто-детект параметров
    actual_obs = weights['fc1.weight'].shape[1]
    actual_hidden = weights['fc1.weight'].shape[0]
    has_fc3 = 'fc3.weight' in weights
    
    print(f"📦 Файл: {os.path.basename(checkpoint_path)}")
    print(f"🔍 Параметры: In={actual_obs}, Hidden={actual_hidden}, Layers={'3' if has_fc3 else '2'}")
    
    model = UniversalTorchNet(obs_dim=actual_obs, hidden_dim=actual_hidden, use_fc3=has_fc3)
    
    # 2. Маппинг весов (MLX -> PyTorch)
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(weights['fc1.weight']))
        model.fc1.bias.copy_(torch.from_numpy(weights['fc1.bias']))
        model.fc2.weight.copy_(torch.from_numpy(weights['fc2.weight']))
        model.fc2.bias.copy_(torch.from_numpy(weights['fc2.bias']))
        
        if has_fc3:
            model.fc3.weight.copy_(torch.from_numpy(weights['fc3.weight']))
            model.fc3.bias.copy_(torch.from_numpy(weights['fc3.bias']))
            
        model.policy_head.weight.copy_(torch.from_numpy(weights['policy_head.weight']))
        model.policy_head.bias.copy_(torch.from_numpy(weights['policy_head.bias']))
        
        if 'value_head.weight' in weights:
            model.value_head.weight.copy_(torch.from_numpy(weights['value_head.weight']))
            model.value_head.bias.copy_(torch.from_numpy(weights['value_head.bias']))

    # 3. Экспорт в ONNX
    onnx_name = os.path.basename(checkpoint_path).replace(".npz", ".onnx")
    dummy_input = torch.randn(1, actual_obs)
    
    torch.onnx.export(model, dummy_input, onnx_name, 
                      input_names=['observation'], output_names=['logits'],
                      dynamic_axes={'observation': {0: 'batch_size'}})
    
    print(f"✅ Экспортировано в: {onnx_name}\n")

if __name__ == "__main__":
    # Список моделей для экспорта (укажи свои пути)
    models_to_convert = [ # Твой чемпион
        "checkpoints/OnlyVersusRandomBiggest.npz"  # Твой Easy-бот (621in)
    ]
    
    for path in models_to_convert:
        if os.path.exists(path):
            export(path)
        else:
            print(f"❌ Файл не найден: {path}")